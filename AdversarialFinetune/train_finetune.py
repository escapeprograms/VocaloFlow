"""Main entry point for adversarial flow-matching fine-tuning of VocaloFlow.

Usage (from AdversarialFinetune/):
    python train_finetune.py --name exp2-v1 --config configs/exp2_v1.yaml

    # Resume:
    python train_finetune.py --resume exp2-v1
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import glob
import os

import torch
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# ── Path setup ─────────────────────────────────────────────────────────────
#
# AdversarialFinetune uses a *flat* module layout (no subpackages beyond
# ``ft_utils/``) specifically to avoid namespace collisions with VocaloFlow's
# ``configs``/``training``/``utils``/``model``/``inference`` packages.  We
# put VocaloFlow first on sys.path (via ft_utils) so its regular package
# imports resolve cleanly.  AdversarialPostnet files live under its own
# ``model/``/``training/`` subpackages which collide with VocaloFlow's, so
# we pull them in via importlib under unique module names.
# ───────────────────────────────────────────────────────────────────────────

from ft_utils import (
    ADV_POSTNET_DIR,
    derive_run_paths,
    import_from_path,
    setup_vocaloflow_sys_path,
    timestamp,
    unpack_batch,
    unpack_optional_features,
)

setup_vocaloflow_sys_path()

# VocaloFlow imports — resolve via sys.path.
from configs.default import VocaloFlowConfig          # noqa: E402
from model.vocaloflow import VocaloFlow               # noqa: E402
from utils.config_utils import rebuild_dataclass_tolerant  # noqa: E402
from utils.dataset import VocaloFlowDataset           # noqa: E402
from utils.collate import vocaloflow_collate_fn, validate_batch_signals  # noqa: E402
from utils.data_helpers import (                      # noqa: E402
    load_manifest, filter_manifest, split_by_song, split_random,
)
from training.lr_schedule import get_lr               # noqa: E402

# AdversarialPostnet imports — via importlib to avoid VocaloFlow/training
# namespace collision.
_disc_mod = import_from_path(
    "ap_discriminator",
    os.path.join(ADV_POSTNET_DIR, "model", "discriminator.py"),
)
PatchDiscriminator = _disc_mod.PatchDiscriminator

_losses_mod = import_from_path(
    "ap_losses",
    os.path.join(ADV_POSTNET_DIR, "training", "losses.py"),
)
masked_l1 = _losses_mod.masked_l1
hinge_d_loss = _losses_mod.hinge_d_loss
hinge_g_loss = _losses_mod.hinge_g_loss
feature_matching_loss = _losses_mod.feature_matching_loss

_crop_mod = import_from_path(
    "ap_random_crop",
    os.path.join(ADV_POSTNET_DIR, "training", "random_crop.py"),
)
CropSpec = _crop_mod.CropSpec
extract_random_crops = _crop_mod.extract_random_crops

# Our own flat-layout modules — unique names, no collision.
from finetune_config import FinetuneConfig            # noqa: E402
from cfm_loss import build_cfm_loss                   # noqa: E402
from ft_checkpoint import (                           # noqa: E402
    extract_checkpoint_step, find_latest_checkpoint,
    load_checkpoint, save_checkpoint,
)
from ode_unroll import unroll_ode                     # noqa: E402
from evaluate_ft import evaluate_inference, validate_recon_4step  # noqa: E402
from dit_discriminator import DiTDiscriminator         # noqa: E402
from disc_augmentation import augment_mel             # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
# Pretrained checkpoint loading
# ═══════════════════════════════════════════════════════════════════════════

def _find_pretrained_checkpoint(config: FinetuneConfig) -> str:
    """Resolve the VocaloFlow checkpoint to load based on config.

    If ``pretrained_step == 0``, pick the latest ``checkpoint_*.pt`` in the
    ``pretrained_run`` directory; otherwise pick the exact step.
    """
    vf_ckpt_dir = os.path.join(
        config.pretrained_vocaloflow_dir, "checkpoints", config.pretrained_run,
    )
    if not os.path.isdir(vf_ckpt_dir):
        raise FileNotFoundError(
            f"Pretrained run directory not found: {vf_ckpt_dir}"
        )

    if config.pretrained_step > 0:
        path = os.path.join(vf_ckpt_dir, f"checkpoint_{config.pretrained_step}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {path}")
        return path

    paths = glob.glob(os.path.join(vf_ckpt_dir, "checkpoint_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoints in {vf_ckpt_dir}")

    return max(paths, key=extract_checkpoint_step)


# ═══════════════════════════════════════════════════════════════════════════
# Warmup helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_effective_lambda(ft_step: int, config: FinetuneConfig, base_lambda: float) -> float:
    """Apply the disc-warmup + linear-ramp schedule to a base lambda.

    Returns 0 for ``ft_step < disc_warmup_steps``, then ramps linearly to
    ``base_lambda`` over the next ``adv_ramp_steps`` steps.
    """
    if ft_step < config.disc_warmup_steps:
        return 0.0
    progress = (ft_step - config.disc_warmup_steps) / max(1, config.adv_ramp_steps)
    return base_lambda * min(1.0, progress)


def get_lambda_cfm(ft_step: int, config: FinetuneConfig) -> float:
    """Linearly decay lambda_cfm from ``lambda_cfm`` to ``lambda_cfm_final``."""
    if config.lambda_cfm == config.lambda_cfm_final:
        return config.lambda_cfm
    progress = min(1.0, ft_step / max(1, config.total_finetune_steps))
    return config.lambda_cfm + (config.lambda_cfm_final - config.lambda_cfm) * progress


# ═══════════════════════════════════════════════════════════════════════════
# Validation (cheap, per val_every) — single-step CFM loss on full val set
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def validate_cfm(
    ema_model: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """CFM velocity-loss validation on the EMA model.  Identical to VocaloFlow."""
    ema_model.eval()
    criterion.eval()
    vel_sum = 0.0
    n = 0
    try:
        for batch in val_loader:
            x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(batch, device)
            optional = unpack_optional_features(batch, device)
            losses = criterion(
                ema_model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask,
                plbert_features=optional.get("plbert_features"),
                speaker_embedding=optional.get("speaker_embedding"),
            )
            vel_sum += losses["velocity"].item()
            n += 1
    finally:
        criterion.train()
    return {"val/velocity_mse": vel_sum / max(1, n)}


# ═══════════════════════════════════════════════════════════════════════════
# Setup helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detect_resume(
    config: FinetuneConfig, device: torch.device,
) -> tuple[dict | None, str | None]:
    """If a checkpoint exists for ``config.run_name``, return (ckpt_dict, wandb_run_id)."""
    if not config.run_name:
        return None, None
    latest = find_latest_checkpoint(config.checkpoint_dir)
    if not latest:
        return None, None
    resume_ckpt = load_checkpoint(latest, device)
    print(f"[{timestamp()}] "
          f"Resuming fine-tune at global_step {resume_ckpt['step']}")
    return resume_ckpt, resume_ckpt.get("wandb_run_id")


def _init_wandb(config: FinetuneConfig, wandb_run_id: str | None) -> None:
    """Initialise wandb for this run (fresh or resume)."""
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    wandb_kwargs = dict(
        entity="archimedesli",
        project="VocaloFlow-AdvFinetune",
        config=vars(config),
        dir=config.log_dir,
    )
    if config.run_name:
        wandb_kwargs["name"] = config.run_name
    if wandb_run_id:
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)

    # wandb autogenerated a name — propagate to config and paths.
    if not config.run_name:
        config.run_name = wandb.run.name
        derive_run_paths(config)
        os.makedirs(config.checkpoint_dir, exist_ok=True)


def _save_config_snapshot(config: FinetuneConfig) -> None:
    """Write ``configs/<run_name>/config.yaml`` on first launch (no-op on resume)."""
    config_dir = os.path.join("./configs", config.run_name)
    config_path = os.path.join(config_dir, "config.yaml")
    if not os.path.exists(config_path):
        os.makedirs(config_dir, exist_ok=True)
        config.to_yaml(config_path)
        print(f"[{timestamp()}] Saved config: {config_path}")


def _build_dataloaders(
    config: FinetuneConfig,
    vf_config: VocaloFlowConfig | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Construct train and val DataLoaders from the manifest."""
    manifest = load_manifest(config.manifest_path, config.data_dir)
    manifest = filter_manifest(manifest, max_dtw_cost=config.max_dtw_cost)
    if config.split_mode == "song":
        train_df, val_df = split_by_song(manifest, config.val_fraction, config.seed)
    elif config.split_mode == "random":
        train_df, val_df = split_random(manifest, config.val_fraction, config.seed)
    else:
        raise ValueError(f"Unknown split_mode: {config.split_mode!r}")

    use_plbert = vf_config.use_plbert if vf_config is not None else False
    use_speaker = config.use_speaker_embedding
    global_spk_path = vf_config.global_speaker_embedding_path if vf_config is not None else ""

    train_ds = VocaloFlowDataset(
        train_df, config.data_dir, config.max_seq_len, training=True,
        use_plbert=use_plbert, use_speaker_embedding=use_speaker,
        global_speaker_embedding_path=global_spk_path,
    )
    val_ds = VocaloFlowDataset(
        val_df, config.data_dir, config.max_seq_len, training=False,
        use_plbert=use_plbert, use_speaker_embedding=use_speaker,
        global_speaker_embedding_path=global_spk_path,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, collate_fn=vocaloflow_collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=vocaloflow_collate_fn,
    )
    return train_loader, val_loader


def _build_optimizers(
    config: FinetuneConfig,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    hf_discriminator: torch.nn.Module | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.Optimizer | None]:
    """Construct generator, discriminator, and optional HF discriminator optimizers."""
    opt_g = torch.optim.AdamW(
        model.parameters(),
        lr=config.gen_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    opt_d = torch.optim.AdamW(
        discriminator.parameters(),
        lr=config.disc_learning_rate,
        betas=(config.adam_beta1, config.adam_beta2),
        weight_decay=config.weight_decay,
    )
    opt_d_hf = None
    if hf_discriminator is not None:
        opt_d_hf = torch.optim.AdamW(
            hf_discriminator.parameters(),
            lr=config.hf_disc_learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay,
        )
    return opt_g, opt_d, opt_d_hf


def _restore_state(
    config: FinetuneConfig,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    resume_ckpt: dict | None,
    pt_ckpt: dict,
    pretrained_global_step: int,
    hf_discriminator: torch.nn.Module | None = None,
    opt_d_hf: torch.optim.Optimizer | None = None,
) -> tuple[int, int]:
    """Load weights into G/D/EMA/optimizers and return (global_step, start_global_step)."""
    if resume_ckpt is not None:
        # Resuming an adversarial run — everything comes from our own ckpt.
        model.load_state_dict(resume_ckpt["model_state_dict"])
        ema_model.load_state_dict(resume_ckpt["ema_model_state_dict"])
        discriminator.load_state_dict(resume_ckpt["discriminator_state_dict"])
        opt_g.load_state_dict(resume_ckpt["opt_g_state_dict"])
        opt_d.load_state_dict(resume_ckpt["opt_d_state_dict"])
        # HF discriminator: load if present, else keep random init (backwards compat).
        if hf_discriminator is not None and "hf_discriminator_state_dict" in resume_ckpt:
            try:
                hf_discriminator.load_state_dict(resume_ckpt["hf_discriminator_state_dict"])
            except RuntimeError:
                print(f"[{timestamp()}] HF discriminator architecture changed — starting from random init")
        if opt_d_hf is not None and "opt_d_hf_state_dict" in resume_ckpt:
            opt_d_hf.load_state_dict(resume_ckpt["opt_d_hf_state_dict"])
        global_step = int(resume_ckpt["step"])
        start_global_step = global_step - _infer_elapsed_ft_steps(
            config.checkpoint_dir, global_step,
        )
        return global_step, start_global_step

    # Fresh fine-tune run: initialise from pretrained VocaloFlow.
    # Use strict=False to allow new modules (e.g. speaker_proj) that don't
    # exist in the pretrained checkpoint.  Zero-init ensures they have no
    # effect at step 0.
    missing_g, unexpected_g = model.load_state_dict(pt_ckpt["model_state_dict"], strict=False)
    missing_e, unexpected_e = ema_model.load_state_dict(pt_ckpt["ema_model_state_dict"], strict=False)
    if unexpected_g:
        print(f"[{timestamp()}] WARNING: unexpected keys in pretrained checkpoint: {unexpected_g}")
    if missing_g:
        print(f"[{timestamp()}] New generator keys (zero-init): {missing_g}")
    if missing_e:
        print(f"[{timestamp()}] New EMA keys (zero-init): {missing_e}")
    # Discriminator stays at random init (intentional — see plan §4.3).
    if not config.reset_gen_optimizer and "optimizer_state_dict" in pt_ckpt:
        # Spec path: load pretrained Adam moments.  Not recommended — see
        # plan §3 for why reset_gen_optimizer=True is the default.
        opt_g.load_state_dict(pt_ckpt["optimizer_state_dict"])
        print(f"[{timestamp()}] "
              f"Loaded pretrained generator optimizer state "
              f"(reset_gen_optimizer=False).")
    global_step = pretrained_global_step if config.preserve_global_step else 0
    return global_step, global_step


def _write_start_step_marker(checkpoint_dir: str, start_global_step: int) -> None:
    """Sidecar file used by ``_infer_elapsed_ft_steps`` on resume."""
    start_marker = os.path.join(checkpoint_dir, "_ft_start_step.txt")
    if not os.path.exists(start_marker):
        with open(start_marker, "w") as f:
            f.write(str(start_global_step))


# ═══════════════════════════════════════════════════════════════════════════
# Core training step
# ═══════════════════════════════════════════════════════════════════════════

def _discriminator_step(
    crop_pairs: list,
    discriminator: torch.nn.Module,
    opt_d: torch.optim.Optimizer,
    config: FinetuneConfig,
    device: torch.device,
) -> tuple[float, float, float]:
    """One optimizer step on the discriminator over all crop scales.

    Returns ``(d_loss_val, d_real_mean, d_fake_mean)`` for logging.
    """
    d_loss = torch.zeros((), device=device)
    d_real_mean = 0.0
    d_fake_mean = 0.0
    for real_crop, fake_crop in crop_pairs:
        real_scores, _ = discriminator(real_crop)
        fake_scores, _ = discriminator(fake_crop.detach())
        d_loss = d_loss + hinge_d_loss(real_scores, fake_scores)
        d_real_mean += real_scores.mean().item()
        d_fake_mean += fake_scores.mean().item()
    d_loss = d_loss / len(crop_pairs)
    d_real_mean /= len(crop_pairs)
    d_fake_mean /= len(crop_pairs)

    opt_d.zero_grad()
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
    opt_d.step()
    return d_loss.item(), d_real_mean, d_fake_mean


def _generator_adv_losses(
    crop_pairs: list,
    discriminator: torch.nn.Module,
    config: FinetuneConfig,
    eff_lambda_fm: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adversarial + feature-matching losses averaged across crop scales."""
    g_adv = torch.zeros((), device=device)
    l_fm = torch.zeros((), device=device)
    for real_crop, fake_crop in crop_pairs:
        fake_scores, fake_feats = discriminator(fake_crop)
        g_adv = g_adv + hinge_g_loss(fake_scores)
        if config.enable_feature_matching and eff_lambda_fm > 0:
            _, real_feats = discriminator(real_crop)
            l_fm = l_fm + feature_matching_loss(real_feats, fake_feats)
    g_adv = g_adv / len(crop_pairs)
    l_fm = l_fm / len(crop_pairs)
    return g_adv, l_fm


# ── DiT discriminator path (full-sequence, no crops) ──────────────────────

def _dit_discriminator_step(
    x_1: torch.Tensor,
    x_1_hat: torch.Tensor,
    padding_mask: torch.Tensor,
    discriminator: torch.nn.Module,
    opt_d: torch.optim.Optimizer,
    config: FinetuneConfig,
    device: torch.device,
    ft_step: int = 0,
    plbert_features: torch.Tensor | None = None,
    speaker_embedding: torch.Tensor | None = None,
) -> tuple[float, float, float]:
    """One discriminator optimizer step on full mel sequences.

    Returns ``(d_loss_val, d_real_mean, d_fake_mean)`` for logging.
    """
    x_1_d = x_1
    x_1_hat_d = x_1_hat.detach()
    plbert_d = plbert_features
    if config.enable_disc_augmentation:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(ft_step * 2)
        x_1_d = augment_mel(x_1, config, rng)
        rng.manual_seed(ft_step * 2)
        x_1_hat_d = augment_mel(x_1_hat.detach(), config, rng)
        if plbert_features is not None:
            rng.manual_seed(ft_step * 2)
            plbert_d = augment_mel(plbert_features, config, rng)

    real_scores, _ = discriminator(x_1_d, padding_mask,
                                   speaker_embedding=speaker_embedding,
                                   plbert_features=plbert_d)
    fake_scores, _ = discriminator(x_1_hat_d, padding_mask,
                                   speaker_embedding=speaker_embedding,
                                   plbert_features=plbert_d)
    d_loss = hinge_d_loss(real_scores, fake_scores)

    if config.logit_center_weight > 0:
        d_loss = d_loss + config.logit_center_weight * real_scores.mean() ** 2

    d_real_mean = real_scores.mean().item()
    d_fake_mean = fake_scores.mean().item()

    opt_d.zero_grad()
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip)
    opt_d.step()
    return d_loss.item(), d_real_mean, d_fake_mean


def _dit_generator_adv_losses(
    x_1: torch.Tensor,
    x_1_hat: torch.Tensor,
    padding_mask: torch.Tensor,
    discriminator: torch.nn.Module,
    config: FinetuneConfig,
    eff_lambda_fm: float,
    device: torch.device,
    ft_step: int = 0,
    plbert_features: torch.Tensor | None = None,
    speaker_embedding: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Adversarial + feature-matching losses on full mel sequences."""
    x_1_g = x_1
    x_1_hat_g = x_1_hat
    plbert_g = plbert_features
    if config.enable_disc_augmentation:
        rng = torch.Generator(device="cpu")
        rng.manual_seed(ft_step * 2 + 1)
        x_1_g = augment_mel(x_1, config, rng)
        rng.manual_seed(ft_step * 2 + 1)
        x_1_hat_g = augment_mel(x_1_hat, config, rng)
        if plbert_features is not None:
            rng.manual_seed(ft_step * 2 + 1)
            plbert_g = augment_mel(plbert_features, config, rng)

    fake_scores, fake_feats = discriminator(x_1_hat_g, padding_mask,
                                            speaker_embedding=speaker_embedding,
                                            plbert_features=plbert_g)
    g_adv = hinge_g_loss(fake_scores)
    l_fm = torch.zeros((), device=device)
    if config.enable_feature_matching and eff_lambda_fm > 0:
        _, real_feats = discriminator(x_1_g, padding_mask,
                                     speaker_embedding=speaker_embedding,
                                     plbert_features=plbert_g)
        l_fm = feature_matching_loss(real_feats, fake_feats)
    return g_adv, l_fm


# ── High-frequency auxiliary discriminator helpers ────────────────────────


def _hf_discriminator_step(
    x_1: torch.Tensor,
    x_1_hat: torch.Tensor,
    padding_mask: torch.Tensor,
    hf_discriminator: torch.nn.Module,
    opt_d_hf: torch.optim.Optimizer,
    config: FinetuneConfig,
    device: torch.device,
    ft_step: int = 0,
) -> tuple[float, float, float]:
    """One HF DiT discriminator step on the upper mel-frequency band."""
    hf_start = config.hf_disc_hf_start
    x_1_hf = x_1[:, :, hf_start:]
    x_1_hat_hf = x_1_hat.detach()[:, :, hf_start:]

    real_scores, _ = hf_discriminator(x_1_hf, padding_mask)
    fake_scores, _ = hf_discriminator(x_1_hat_hf, padding_mask)
    d_loss = hinge_d_loss(real_scores, fake_scores)

    if config.logit_center_weight > 0:
        d_loss = d_loss + config.logit_center_weight * real_scores.mean() ** 2

    d_real_mean = real_scores.mean().item()
    d_fake_mean = fake_scores.mean().item()

    opt_d_hf.zero_grad()
    d_loss.backward()
    torch.nn.utils.clip_grad_norm_(hf_discriminator.parameters(), config.grad_clip)
    opt_d_hf.step()
    return d_loss.item(), d_real_mean, d_fake_mean


def _hf_generator_adv_loss(
    x_1_hat: torch.Tensor,
    padding_mask: torch.Tensor,
    hf_discriminator: torch.nn.Module,
    config: FinetuneConfig,
    device: torch.device,
    ft_step: int = 0,
) -> torch.Tensor:
    """Adversarial loss from the HF DiT discriminator for the generator."""
    x_1_hat_hf = x_1_hat[:, :, config.hf_disc_hf_start:]
    fake_scores, _ = hf_discriminator(x_1_hat_hf, padding_mask)
    return hinge_g_loss(fake_scores)


def _extra_d_steps(
    d_batch_iter,
    train_loader: DataLoader,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_d: torch.optim.Optimizer,
    config: FinetuneConfig,
    ft_step: int,
    crop_specs: list,
    device: torch.device,
    hf_discriminator: torch.nn.Module | None = None,
    opt_d_hf: torch.optim.Optimizer | None = None,
):
    """Run additional discriminator steps with fresh batches (no generator gradient)."""
    eff_lambda_adv = get_effective_lambda(
        ft_step, config, config.lambda_adv,
    ) if config.enable_adversarial else 0.0
    if eff_lambda_adv <= 0:
        return d_batch_iter

    for extra_idx in range(config.disc_steps_per_gen - 1):
        try:
            batch = next(d_batch_iter)
        except StopIteration:
            d_batch_iter = iter(train_loader)
            batch = next(d_batch_iter)

        x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(batch, device)
        optional = unpack_optional_features(batch, device)
        plbert_features = optional.get("plbert_features")
        speaker_embedding = optional.get("speaker_embedding")

        with torch.no_grad():
            x_1_hat = unroll_ode(
                model, x_0, f0, voicing, phoneme_ids, padding_mask,
                num_steps=config.ode_num_steps,
                method=config.ode_method,
                grad_checkpoint=False,
                plbert_features=plbert_features,
                speaker_embedding=speaker_embedding,
            )

        if config.disc_type == "dit":
            _dit_discriminator_step(
                x_1, x_1_hat, padding_mask,
                discriminator, opt_d, config, device,
                ft_step=ft_step + 1_000_000 * (extra_idx + 1),
                plbert_features=plbert_features,
                speaker_embedding=speaker_embedding,
            )
        else:
            crop_pairs = extract_random_crops(
                x_1, x_1_hat, crop_specs, padding_mask,
            )
            if crop_pairs:
                _discriminator_step(
                    crop_pairs, discriminator, opt_d, config, device,
                )

        if config.use_hf_discriminator and hf_discriminator is not None:
            _hf_discriminator_step(
                x_1, x_1_hat, padding_mask,
                hf_discriminator, opt_d_hf, config, device,
                ft_step=ft_step + 1_000_000 * (extra_idx + 1),
            )

    return d_batch_iter


def _training_step(
    batch: dict,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    opt_g: torch.optim.Optimizer,
    opt_d: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    crop_specs: list,
    config: FinetuneConfig,
    ft_step: int,
    device: torch.device,
    hf_discriminator: torch.nn.Module | None = None,
    opt_d_hf: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One full training step: LR update, forward, D update, G update.

    Returns a dict of scalars suitable for TB/wandb logging.
    """
    x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(batch, device)
    optional = unpack_optional_features(batch, device)
    plbert_features = optional.get("plbert_features")
    speaker_embedding = optional.get("speaker_embedding")

    # Learning rates.  Generator is constant; disc has warmup + cosine.
    lr_g = config.gen_learning_rate
    lr_d = get_lr(
        ft_step, config.disc_lr_warmup_steps,
        config.total_finetune_steps, config.disc_learning_rate,
    )
    for pg in opt_g.param_groups:
        pg["lr"] = lr_g
    for pg in opt_d.param_groups:
        pg["lr"] = lr_d

    lr_d_hf = 0.0
    if opt_d_hf is not None:
        lr_d_hf = get_lr(
            ft_step, config.hf_disc_lr_warmup_steps,
            config.total_finetune_steps, config.hf_disc_learning_rate,
        )
        for pg in opt_d_hf.param_groups:
            pg["lr"] = lr_d_hf

    eff_lambda_adv = get_effective_lambda(
        ft_step, config, config.lambda_adv,
    ) if config.enable_adversarial else 0.0
    eff_lambda_fm = get_effective_lambda(
        ft_step, config, config.lambda_fm,
    ) if config.enable_feature_matching else 0.0
    eff_lambda_hf_adv = get_effective_lambda(
        ft_step, config, config.lambda_hf_adv,
    ) if config.use_hf_discriminator else 0.0

    # === CFM phase ============================================
    if config.enable_cfm:
        cfm_out = criterion(
            model, x_0, x_1, f0, voicing, phoneme_ids, padding_mask,
            plbert_features=plbert_features,
            speaker_embedding=speaker_embedding,
        )
        l_cfm = cfm_out["velocity"]
        eb_std_val = cfm_out["eb_std"].item()
    else:
        l_cfm = torch.zeros((), device=device)
        eb_std_val = 0.0

    # === ODE unroll phase =====================================
    need_ode = (
        config.enable_reconstruction
        or config.enable_adversarial
        or config.enable_feature_matching
        or config.use_hf_discriminator
    )
    if need_ode:
        x_1_hat = unroll_ode(
            model, x_0, f0, voicing, phoneme_ids, padding_mask,
            num_steps=config.ode_num_steps,
            method=config.ode_method,
            grad_checkpoint=config.ode_grad_checkpoint,
            plbert_features=plbert_features,
            speaker_embedding=speaker_embedding,
        )
    else:
        x_1_hat = None

    # === Discriminator update =================================
    d_loss_val = 0.0
    d_real_mean = 0.0
    d_fake_mean = 0.0

    use_dit = config.disc_type == "dit"

    if use_dit:
        # DiT discriminator: full-sequence path (no crops)
        if config.enable_adversarial and x_1_hat is not None:
            d_loss_val, d_real_mean, d_fake_mean = _dit_discriminator_step(
                x_1, x_1_hat, padding_mask,
                discriminator, opt_d, config, device,
                ft_step=ft_step,
                plbert_features=plbert_features,
                speaker_embedding=speaker_embedding,
            )
    else:
        # Patch discriminator: crop-based path
        if x_1_hat is not None:
            crop_pairs = extract_random_crops(
                x_1, x_1_hat, crop_specs, padding_mask,
            )
        else:
            crop_pairs = []
        if config.enable_adversarial and crop_pairs:
            d_loss_val, d_real_mean, d_fake_mean = _discriminator_step(
                crop_pairs, discriminator, opt_d, config, device,
            )

    # === HF discriminator update ==============================
    d_hf_loss_val = 0.0
    d_hf_real_mean = 0.0
    d_hf_fake_mean = 0.0
    if (config.use_hf_discriminator and hf_discriminator is not None
            and x_1_hat is not None):
        d_hf_loss_val, d_hf_real_mean, d_hf_fake_mean = _hf_discriminator_step(
            x_1, x_1_hat, padding_mask,
            hf_discriminator, opt_d_hf, config, device,
            ft_step=ft_step,
        )

    # === Generator losses ====================================
    l_rec = torch.zeros((), device=device)
    g_adv = torch.zeros((), device=device)
    l_fm = torch.zeros((), device=device)

    if config.enable_reconstruction and x_1_hat is not None:
        l_rec = masked_l1(x_1_hat, x_1, padding_mask)

    if config.enable_adversarial and x_1_hat is not None and eff_lambda_adv > 0:
        if use_dit:
            g_adv, l_fm = _dit_generator_adv_losses(
                x_1, x_1_hat, padding_mask,
                discriminator, config, eff_lambda_fm, device,
                ft_step=ft_step,
                plbert_features=plbert_features,
                speaker_embedding=speaker_embedding,
            )
        else:
            if crop_pairs:
                g_adv, l_fm = _generator_adv_losses(
                    crop_pairs, discriminator, config, eff_lambda_fm, device,
                )

    g_adv_hf = torch.zeros((), device=device)
    if (config.use_hf_discriminator and hf_discriminator is not None
            and x_1_hat is not None and eff_lambda_hf_adv > 0):
        g_adv_hf = _hf_generator_adv_loss(
            x_1_hat, padding_mask,
            hf_discriminator, config, device,
            ft_step=ft_step,
        )

    current_lambda_cfm = get_lambda_cfm(ft_step, config)
    adv_grad_norm = 0.0
    cfm_grad_norm = 0.0
    hf_adv_grad_norm = 0.0

    use_grad_norm = (
        config.enable_grad_norm
        and config.enable_adversarial
        and eff_lambda_adv > 0
    )

    if use_grad_norm:
        # === Gradient normalization path ===
        # Phase 1: CFM + rec backward — accumulate grads into p.grad
        non_adv_loss = current_lambda_cfm * l_cfm + config.lambda_rec * l_rec
        opt_g.zero_grad()
        non_adv_loss.backward(retain_graph=True)

        cfm_grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float("inf"),
        ).item()

        # Phase 2: DiT adversarial grads (independently L2-normalized)
        need_hf_grad = config.use_hf_discriminator and eff_lambda_hf_adv > 0
        adv_loss = eff_lambda_adv * g_adv + eff_lambda_fm * l_fm
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        adv_grads = torch.autograd.grad(
            adv_loss, trainable_params,
            retain_graph=need_hf_grad, allow_unused=True,
        )
        adv_grad_tensors = [
            g if g is not None else torch.zeros_like(p)
            for g, p in zip(adv_grads, trainable_params)
        ]
        adv_grad_norm = sum(
            (g ** 2).sum() for g in adv_grad_tensors
        ).sqrt().item()

        if adv_grad_norm > 1e-8:
            scale = 1.0 / adv_grad_norm
            for p, ag in zip(trainable_params, adv_grad_tensors):
                if p.grad is not None:
                    p.grad.add_(ag, alpha=scale)

        # Phase 3: HF adversarial grads (independently L2-normalized)
        if need_hf_grad:
            hf_adv_loss = eff_lambda_hf_adv * g_adv_hf
            hf_adv_grads = torch.autograd.grad(
                hf_adv_loss, trainable_params,
                retain_graph=False, allow_unused=True,
            )
            hf_adv_grad_tensors = [
                g if g is not None else torch.zeros_like(p)
                for g, p in zip(hf_adv_grads, trainable_params)
            ]
            hf_adv_grad_norm = sum(
                (g ** 2).sum() for g in hf_adv_grad_tensors
            ).sqrt().item()

            if hf_adv_grad_norm > 1e-8:
                hf_scale = 1.0 / hf_adv_grad_norm
                for p, ag in zip(trainable_params, hf_adv_grad_tensors):
                    if p.grad is not None:
                        p.grad.add_(ag, alpha=hf_scale)

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt_g.step()
        total_g_val = non_adv_loss.item() + adv_loss.item() + (
            (eff_lambda_hf_adv * g_adv_hf).item() if need_hf_grad else 0.0
        )
    else:
        # === Original single-backward path ===
        total_g = (
            current_lambda_cfm * l_cfm
            + config.lambda_rec * l_rec
            + eff_lambda_adv * g_adv
            + eff_lambda_fm * l_fm
            + eff_lambda_hf_adv * g_adv_hf
        )
        opt_g.zero_grad()
        total_g.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        opt_g.step()
        total_g_val = total_g.item()

    return {
        "train/loss_total_g": total_g_val,
        "train/loss_cfm": l_cfm.item(),
        "train/loss_rec": l_rec.item(),
        "train/loss_g_adv": g_adv.item(),
        "train/loss_fm": l_fm.item(),
        "train/loss_d": d_loss_val,
        "train/lambda_adv": eff_lambda_adv,
        "train/lambda_fm": eff_lambda_fm,
        "train/lambda_cfm": current_lambda_cfm,
        "train/adv_grad_norm": adv_grad_norm,
        "train/cfm_grad_norm": cfm_grad_norm,
        "train/lr_g": lr_g,
        "train/lr_d": lr_d,
        "train/d_real_mean": d_real_mean,
        "train/d_fake_mean": d_fake_mean,
        "train/eb_weight_std": eb_std_val,
        "train/loss_d_hf": d_hf_loss_val,
        "train/loss_g_adv_hf": g_adv_hf.item(),
        "train/lambda_hf_adv": eff_lambda_hf_adv,
        "train/hf_adv_grad_norm": hf_adv_grad_norm,
        "train/lr_d_hf": lr_d_hf,
        "train/d_hf_real_mean": d_hf_real_mean,
        "train/d_hf_fake_mean": d_hf_fake_mean,
    }

    # Projection layer gradient norms (speaker / PL-BERT conditioning)
    for attr, key in [
        ("speaker_proj", "train/gen_speaker_proj_grad_norm"),
        ("plbert_proj", "train/gen_plbert_proj_grad_norm"),
    ]:
        mod = getattr(model, attr, None)
        if mod is not None and hasattr(mod, "weight") and mod.weight.grad is not None:
            metrics[key] = mod.weight.grad.norm().item()

    for attr, key in [
        ("speaker_proj", "train/disc_speaker_proj_grad_norm"),
        ("plbert_proj", "train/disc_plbert_proj_grad_norm"),
    ]:
        mod = getattr(discriminator, attr, None)
        if mod is not None and hasattr(mod, "weight") and mod.weight.grad is not None:
            metrics[key] = mod.weight.grad.norm().item()


def _ema_update(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    global_step: int,
    config: FinetuneConfig,
) -> None:
    """Warmup-safe EMA update.

    The decay formula ``min(ema_decay, (step+1)/(step+10))`` gives 0.1 at step=0,
    which would overwrite the pretrained EMA in a single update.  Callers rely
    on ``preserve_global_step=True`` (default) to keep ``global_step`` large
    enough that decay stays saturated at ``ema_decay``.
    """
    effective_decay = min(
        config.ema_decay,
        (global_step + 1) / (global_step + 10),
    )
    with torch.no_grad():
        for p_ema, p_model in zip(ema_model.parameters(), model.parameters()):
            p_ema.lerp_(p_model, 1.0 - effective_decay)


# ═══════════════════════════════════════════════════════════════════════════
# Periodic helpers: logging, validation, evaluation, checkpointing
# ═══════════════════════════════════════════════════════════════════════════

def _log_train_metrics(
    writer: SummaryWriter,
    metrics: dict[str, float],
    global_step: int,
    ft_step: int,
) -> None:
    """Mirror metrics to TensorBoard + wandb, and print a one-line summary."""
    metrics = {**metrics, "train/ft_step": ft_step}
    for k, v in metrics.items():
        writer.add_scalar(k, v, global_step)
    wandb.log(metrics, step=global_step)
    print(
        f"[{timestamp()}] "
        f"step={global_step} (ft={ft_step}) "
        f"cfm={metrics['train/loss_cfm']:.4f} "
        f"rec={metrics['train/loss_rec']:.4f} "
        f"g_adv={metrics['train/loss_g_adv']:.4f} "
        f"fm={metrics['train/loss_fm']:.4f} "
        f"d={metrics['train/loss_d']:.4f} "
        f"D(r)={metrics['train/d_real_mean']:.3f} "
        f"D(f)={metrics['train/d_fake_mean']:.3f} "
        f"λa={metrics['train/lambda_adv']:.4f} "
        f"λcfm={metrics.get('train/lambda_cfm', 0):.3f} "
        f"adv_gn={metrics.get('train/adv_grad_norm', 0):.2e} "
        f"cfm_gn={metrics.get('train/cfm_grad_norm', 0):.2e}"
        + (f" d_hf={metrics.get('train/loss_d_hf', 0):.4f}"
           f" g_hf={metrics.get('train/loss_g_adv_hf', 0):.4f}"
           f" D_hf(r)={metrics.get('train/d_hf_real_mean', 0):.3f}"
           f" D_hf(f)={metrics.get('train/d_hf_fake_mean', 0):.3f}"
           if metrics.get("train/lambda_hf_adv", 0) > 0 else "")
    )


def _run_validation(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.Module,
    writer: SummaryWriter,
    device: torch.device,
    global_step: int,
) -> None:
    """Run CFM + 4-step reconstruction validation, log, flip models back to train()."""
    val_metrics = validate_cfm(ema_model, val_loader, criterion, device)
    # 4-step Euler reconstruction L1 on the full val set.  Cheap
    # relative to training: ~4x one val CFM pass, still O(minutes).
    val_metrics.update(validate_recon_4step(ema_model, val_loader, device))
    for k, v in val_metrics.items():
        writer.add_scalar(k, v, global_step)
    wandb.log(val_metrics, step=global_step)
    print(
        f"[{timestamp()}] "
        f"step={global_step} "
        f"val_vel={val_metrics['val/velocity_mse']:.4f} "
        f"val_rec4={val_metrics['val/l1_reconstruction_4step']:.4f}"
    )
    model.train()
    discriminator.train()


def _run_heavy_evaluation(
    ema_model: torch.nn.Module,
    model: torch.nn.Module,
    discriminator: torch.nn.Module,
    val_loader: DataLoader,
    config: FinetuneConfig,
    writer: SummaryWriter,
    device: torch.device,
    global_step: int,
) -> None:
    """32-step-vs-4-step inference eval + optional mel-plot dump."""
    plots_dir = (
        os.path.join(config.log_dir, "mels", f"step_{global_step}")
        if config.save_mel_plots else None
    )
    eval_metrics = evaluate_inference(
        ema_model, val_loader, device,
        num_samples=config.eval_num_samples,
        save_plots_dir=plots_dir,
        log_plots_to_wandb=config.save_mel_plots,
        wandb_step=global_step,
        num_plots=config.num_mel_plots,
    )
    for k, v in eval_metrics.items():
        writer.add_scalar(k, v, global_step)
    wandb.log(eval_metrics, step=global_step)
    print(
        f"[{timestamp()}] "
        f"step={global_step} "
        f"l1_32mid={eval_metrics['eval/l1_32step_midpoint']:.4f} "
        f"l1_4eul={eval_metrics['eval/l1_4step_euler']:.4f} "
        f"ratio={eval_metrics['eval/l1_ratio']:.3f}"
        + (f"  [mels -> {plots_dir}]" if plots_dir else "")
    )
    model.train()
    discriminator.train()


# ═══════════════════════════════════════════════════════════════════════════
# Main training
# ═══════════════════════════════════════════════════════════════════════════

def train(config: FinetuneConfig) -> None:
    """Run adversarial fine-tune training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{timestamp()}] Device: {device}")

    # ── Setup: paths, resume detection, wandb, config snapshot ────────────
    derive_run_paths(config)
    resume_ckpt, wandb_run_id = _detect_resume(config, device)
    _init_wandb(config, wandb_run_id)
    _save_config_snapshot(config)

    # ── Load pretrained VocaloFlow ────────────────────────────────────────
    pt_ckpt_path = _find_pretrained_checkpoint(config)
    print(f"[{timestamp()}] Pretrained checkpoint: {pt_ckpt_path}")
    pt_ckpt = torch.load(pt_ckpt_path, map_location=device, weights_only=False)
    vf_config = rebuild_dataclass_tolerant(pt_ckpt["config"], VocaloFlowConfig)
    pretrained_global_step = int(pt_ckpt["step"])

    # ── Models, data, optimizers ──────────────────────────────────────────
    model = VocaloFlow(vf_config).to(device)
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[{timestamp()}] VocaloFlow parameters: {param_count:,}")

    train_loader, val_loader = _build_dataloaders(config, vf_config)

    _probe = next(iter(train_loader))
    validate_batch_signals(
        _probe,
        expect_plbert=vf_config.use_plbert,
        expect_speaker_embedding=config.use_speaker_embedding,
    )
    del _probe

    if config.disc_type == "dit":
        discriminator = DiTDiscriminator(
            mel_dim=vf_config.mel_channels,
            hidden_dim=config.disc_dit_hidden_dim,
            num_blocks=config.disc_dit_num_blocks,
            num_heads=config.disc_dit_num_heads,
            ffn_dim=config.disc_dit_ffn_dim,
            max_len=vf_config.max_seq_len + 1,
            feature_block_indices=list(config.disc_dit_feature_blocks),
            use_plbert_input=config.disc_use_plbert_input,
            plbert_feature_dim=vf_config.plbert_feature_dim,
            disc_plbert_proj_dim=config.disc_plbert_proj_dim,
            use_speaker_input=config.disc_use_speaker_input,
            speaker_embedding_dim=vf_config.speaker_embedding_dim,
            disc_speaker_proj_dim=config.disc_speaker_proj_dim,
        ).to(device)
    else:
        discriminator = PatchDiscriminator(channels=list(config.disc_channels)).to(device)
    disc_params = sum(p.numel() for p in discriminator.parameters())
    print(f"[{timestamp()}] Discriminator ({config.disc_type}): {disc_params:,} params")

    hf_discriminator = None
    if config.use_hf_discriminator:
        hf_mel_dim = vf_config.mel_channels - config.hf_disc_hf_start
        hf_discriminator = DiTDiscriminator(
            mel_dim=hf_mel_dim,
            hidden_dim=config.hf_disc_hidden_dim,
            num_blocks=config.hf_disc_num_blocks,
            num_heads=config.hf_disc_num_heads,
            ffn_dim=config.hf_disc_ffn_dim,
            max_len=vf_config.max_seq_len + 1,
            feature_block_indices=list(config.hf_disc_feature_blocks),
        ).to(device)
        hf_params = sum(p.numel() for p in hf_discriminator.parameters())
        print(f"[{timestamp()}] HF Discriminator (DiT): {hf_params:,} params")

    opt_g, opt_d, opt_d_hf = _build_optimizers(
        config, model, discriminator, hf_discriminator,
    )

    # ── Restore state ─────────────────────────────────────────────────────
    global_step, start_global_step = _restore_state(
        config, model, ema_model, discriminator, opt_g, opt_d,
        resume_ckpt, pt_ckpt, pretrained_global_step,
        hf_discriminator=hf_discriminator, opt_d_hf=opt_d_hf,
    )
    del resume_ckpt, pt_ckpt

    end_global_step = start_global_step + config.total_finetune_steps
    _write_start_step_marker(config.checkpoint_dir, start_global_step)
    print(f"[{timestamp()}] "
          f"global_step range: {start_global_step} -> {end_global_step}")

    # ── CFM criterion + crops ─────────────────────────────────────────────
    criterion = build_cfm_loss(
        cfg_dropout_prob=config.cfg_dropout_prob,
        sigma_min=config.sigma_min,
        energy_balanced_loss=config.energy_balanced_loss,
        energy_balance_lambda=config.energy_balance_lambda,
        energy_balance_epsilon=config.energy_balance_epsilon,
        energy_balance_mode=config.energy_balance_mode,
    )
    criterion.train()
    crop_specs = [CropSpec(t, f) for t, f in config.crop_specs]

    writer = SummaryWriter(log_dir=config.log_dir)

    # ── Training loop ─────────────────────────────────────────────────────
    model.train()
    discriminator.train()
    if hf_discriminator is not None:
        hf_discriminator.train()

    # Separate iterator for extra discriminator steps (fresh batches)
    d_batch_iter = iter(train_loader) if config.disc_steps_per_gen > 1 else None

    while global_step < end_global_step:
        for batch in train_loader:
            if global_step >= end_global_step:
                break

            ft_step = global_step - start_global_step

            metrics = _training_step(
                batch, model, discriminator, opt_g, opt_d,
                criterion, crop_specs, config, ft_step, device,
                hf_discriminator=hf_discriminator, opt_d_hf=opt_d_hf,
            )

            if d_batch_iter is not None and config.disc_steps_per_gen > 1:
                d_batch_iter = _extra_d_steps(
                    d_batch_iter, train_loader,
                    model, discriminator, opt_d,
                    config, ft_step, crop_specs, device,
                    hf_discriminator=hf_discriminator, opt_d_hf=opt_d_hf,
                )

            _ema_update(ema_model, model, global_step, config)

            if global_step % config.log_every == 0:
                _log_train_metrics(writer, metrics, global_step, ft_step)

            if global_step > start_global_step and ft_step % config.val_every == 0:
                _run_validation(
                    ema_model, model, discriminator,
                    val_loader, criterion, writer, device, global_step,
                )
                if hf_discriminator is not None:
                    hf_discriminator.train()

            if global_step > start_global_step and ft_step % config.eval_every == 0:
                _run_heavy_evaluation(
                    ema_model, model, discriminator,
                    val_loader, config, writer, device, global_step,
                )
                if hf_discriminator is not None:
                    hf_discriminator.train()

            if global_step > start_global_step and ft_step % config.save_every == 0:
                save_checkpoint(
                    model, ema_model, discriminator, opt_g, opt_d,
                    global_step, config, vf_config, wandb.run.id,
                    hf_discriminator=hf_discriminator, opt_d_hf=opt_d_hf,
                )

            global_step += 1

    # Final save
    save_checkpoint(
        model, ema_model, discriminator, opt_g, opt_d,
        global_step, config, vf_config, wandb.run.id,
        hf_discriminator=hf_discriminator, opt_d_hf=opt_d_hf,
    )
    writer.close()
    wandb.finish()
    print(f"[{timestamp()}] Fine-tuning complete.")


def _infer_elapsed_ft_steps(checkpoint_dir: str, current_step: int) -> int:
    """Recover the number of fine-tune steps elapsed since this run started.

    Stored in ``_ft_start_step.txt`` alongside the checkpoints so resume does
    not need to re-query the pretrained checkpoint.  Falls back to 0 if the
    marker is missing (keeps resume functional for older runs).
    """
    marker = os.path.join(checkpoint_dir, "_ft_start_step.txt")
    if not os.path.exists(marker):
        return 0
    with open(marker) as f:
        start = int(f.read().strip())
    return current_step - start


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _load_config_from_checkpoint(run_name: str) -> FinetuneConfig:
    """Reconstruct a FinetuneConfig from the latest checkpoint of *run_name*.

    Reads from the ``finetune_config`` key.  Falls back to ``config`` for
    backwards compatibility with any pre-dual-schema checkpoints, but only
    if that value looks like a FinetuneConfig (not a VocaloFlowConfig).
    """
    checkpoint_dir = os.path.join("./checkpoints", run_name)
    latest = find_latest_checkpoint(checkpoint_dir)
    if latest is None:
        raise FileNotFoundError(
            f"No checkpoint found in {checkpoint_dir} — cannot resume {run_name!r}"
        )
    print(f"[{timestamp()}] Loading saved config from {latest}")
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)

    saved = ckpt.get("finetune_config")
    if saved is None:
        # Legacy schema: single "config" key holding the FinetuneConfig.
        legacy = ckpt.get("config")
        if legacy is not None and type(legacy).__name__ == "FinetuneConfig":
            saved = legacy
    if saved is None:
        raise ValueError(
            f"Checkpoint {latest} has no embedded FinetuneConfig — "
            f"cannot safely resume."
        )

    return rebuild_dataclass_tolerant(saved, FinetuneConfig)


def _apply_yaml_overrides(config: FinetuneConfig, path: str) -> None:
    """Merge a YAML file of partial overrides onto an existing config."""
    import yaml

    with open(path) as f:
        overrides = yaml.safe_load(f) or {}

    fields_by_name = {f.name: f for f in dataclasses.fields(config)}
    for key, value in overrides.items():
        if key not in fields_by_name:
            raise ValueError(
                f"Unknown config field in {path}: {key!r}. "
                f"Valid fields: {sorted(fields_by_name)}"
            )
        expected_type = fields_by_name[key].type
        if expected_type is float and isinstance(value, int):
            value = float(value)
        setattr(config, key, value)


def main():
    parser = argparse.ArgumentParser(
        description="Adversarial flow-matching fine-tuning of VocaloFlow",
    )
    parser.add_argument("--name", type=str, default=None,
                        help="Run name (checkpoint/log/config subdir and wandb name)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume training for the given run name")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config overrides")
    parser.add_argument("--pretrained-run", type=str, default=None,
                        help="VocaloFlow run name under ../VocaloFlow/checkpoints/")
    parser.add_argument("--pretrained-step", type=int, default=None,
                        help="Exact pretrained step (0 = latest)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--total-finetune-steps", type=int, default=None)
    args = parser.parse_args()

    if args.resume:
        config = _load_config_from_checkpoint(args.resume)
        config.run_name = args.resume
    else:
        config = FinetuneConfig()
        if args.name:
            config.run_name = args.name

    if args.config:
        _apply_yaml_overrides(config, args.config)
    if args.pretrained_run:
        config.pretrained_run = args.pretrained_run
    if args.pretrained_step is not None:
        config.pretrained_step = args.pretrained_step
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.total_finetune_steps:
        config.total_finetune_steps = args.total_finetune_steps

    train(config)


if __name__ == "__main__":
    main()
