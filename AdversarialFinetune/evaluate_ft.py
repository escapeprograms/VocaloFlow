"""Periodic true-inference evaluation for adversarial fine-tuning.

Unlike the in-loop CFM validation (which uses single-step velocity prediction),
this module runs the full ODE integration to measure quality at both the
inference-time setting (32-step midpoint) and the training-time setting (4-step
Euler).  The gap between the two exposes any distillation-transfer failure:
an adversarial signal that only shapes the 4-step output won't improve the
32-step output.

Mel-only: no vocoding, no WER, no F0 RMSE.  Those live in a separate post-run
script that runs once on the final checkpoint.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ft_utils import VOCALOFLOW_DIR, import_from_path, unpack_batch


# ── Import sample_ode from VocaloFlow (no namespace collision) ────────────
_inf_mod = import_from_path(
    "vf_inference",
    os.path.join(VOCALOFLOW_DIR, "inference", "inference.py"),
)
sample_ode = _inf_mod.sample_ode


def _masked_l1(pred: Tensor, target: Tensor, padding_mask: Tensor | None) -> float:
    """Masked L1 returning a Python float."""
    diff = (pred - target).abs()
    if padding_mask is not None:
        mask = padding_mask.unsqueeze(-1).float()
        val = (diff * mask).sum() / (mask.sum() * diff.shape[-1])
    else:
        val = diff.mean()
    return val.item()


def _build_mel_comparison_figure(
    prior_mel: Tensor,
    target_mel: Tensor,
    pred_32: Tensor,
    pred_4: Tensor,
    length: int,
):
    """Return a 4-panel mel-comparison matplotlib Figure.

    Panels, left to right: prior | target | 32-step midpoint | 4-step Euler.
    All panels share the same colour range so visual differences reflect
    actual mel-value differences, not rescaling.  Each mel is truncated to
    ``length`` valid frames before plotting.

    Inputs are 2D tensors ``(T, 128)`` on any device.  The caller owns the
    returned Figure: it is responsible for saving (``fig.savefig``),
    uploading to wandb (``wandb.Image(fig)``), and releasing
    (``plt.close(fig)``).  This function performs no disk or network I/O.
    """
    # Local import — cheap after the caller has already initialised
    # matplotlib with the Agg backend, and keeps this module importable on
    # machines without matplotlib when plotting is disabled.
    import matplotlib.pyplot as plt

    mels = [
        ("prior", prior_mel),
        ("target", target_mel),
        ("pred 32-step midpoint", pred_32),
        ("pred 4-step Euler", pred_4),
    ]
    # Truncate to valid length and move to CPU numpy.  Transpose so mel bins
    # are on the Y axis (rows) and time is on the X axis (columns).
    arrays = [(name, m[:length].detach().cpu().float().numpy().T) for name, m in mels]

    vmin = min(a.min() for _, a in arrays)
    vmax = max(a.max() for _, a in arrays)

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.2), constrained_layout=True)
    for ax, (name, arr) in zip(axes, arrays):
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("frame")
    axes[0].set_ylabel("mel bin")
    fig.colorbar(im, ax=axes, shrink=0.9, pad=0.01)
    return fig


@torch.no_grad()
def evaluate_inference(
    ema_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_samples: int = 8,
    save_plots_dir: str | None = None,
    log_plots_to_wandb: bool = False,
    wandb_step: int | None = None,
    num_plots: int = 4,
) -> dict[str, float]:
    """Run both inference-setting and training-setting ODEs on a val subset.

    Args:
        ema_model: EMA weights (we always evaluate the EMA, not the live model).
        val_loader: validation DataLoader.
        device: cuda/cpu.
        num_samples: rough upper bound on the number of val items to evaluate.
            Stops at the first batch that pushes the running count past this.
        save_plots_dir: If set, save 4-panel mel comparison PNGs here
            (one per evaluated sample, up to ``num_plots``).  Parent dirs
            created as needed.
        log_plots_to_wandb: If True, also upload the first ``num_plots``
            figures to wandb as ``eval/mel_comparison_<i>`` at ``wandb_step``.
        wandb_step: Step for wandb.log — required if log_plots_to_wandb=True.
        num_plots: Cap on the number of per-sample PNGs / wandb images.

    Returns:
        Dict with keys:
            eval/l1_32step_midpoint  — mel L1 at the inference-time setting.
            eval/l1_4step_euler      — mel L1 at the training-time setting.
            eval/l1_ratio            — 4-step / 32-step (>=1 expected).
    """
    ema_model.eval()
    sum_32 = 0.0
    sum_4 = 0.0
    n = 0
    plots_saved = 0
    wandb_images: dict = {}

    # One-time plotting setup.  Import matplotlib / wandb only when needed,
    # select the Agg backend before any pyplot use, and create the output
    # directory once — inside the loop these would be repeated no-ops but
    # noise.  If plotting is disabled, neither dependency is touched.
    want_plots = (save_plots_dir is not None or log_plots_to_wandb) and num_plots > 0
    plt = None
    wandb = None
    if want_plots:
        import matplotlib as _mpl
        _mpl.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F811
        if save_plots_dir is not None:
            os.makedirs(save_plots_dir, exist_ok=True)
        if log_plots_to_wandb:
            import wandb  # noqa: F811

    for batch in val_loader:
        if n >= num_samples:
            break

        x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(batch, device)
        lengths = batch["length"].tolist() if "length" in batch else [x_0.shape[1]] * x_0.shape[0]
        B = x_0.shape[0]

        mel_32 = sample_ode(
            ema_model, x_0, f0, voicing, phoneme_ids,
            num_steps=32, method="midpoint",
            padding_mask=padding_mask, diagnostics=False, cfg_scale=1.0,
        )
        mel_4 = sample_ode(
            ema_model, x_0, f0, voicing, phoneme_ids,
            num_steps=4, method="euler",
            padding_mask=padding_mask, diagnostics=False, cfg_scale=1.0,
        )

        sum_32 += _masked_l1(mel_32, x_1, padding_mask) * B
        sum_4 += _masked_l1(mel_4, x_1, padding_mask) * B
        n += B

        if want_plots and plots_saved < num_plots:
            for i in range(B):
                if plots_saved >= num_plots:
                    break
                fig = _build_mel_comparison_figure(
                    x_0[i], x_1[i], mel_32[i], mel_4[i],
                    length=max(1, int(lengths[i])),
                )
                if save_plots_dir is not None:
                    fig.savefig(
                        os.path.join(save_plots_dir, f"sample_{plots_saved}.png"),
                        dpi=100,
                    )
                if log_plots_to_wandb:
                    wandb_images[f"eval/mel_comparison_{plots_saved}"] = wandb.Image(fig)
                plt.close(fig)                          # release matplotlib memory
                plots_saved += 1

    if log_plots_to_wandb and wandb_images:
        wandb.log(wandb_images, step=wandb_step)

    n = max(1, n)
    l1_32 = sum_32 / n
    l1_4 = sum_4 / n
    return {
        "eval/l1_32step_midpoint": l1_32,
        "eval/l1_4step_euler": l1_4,
        "eval/l1_ratio": l1_4 / max(l1_32, 1e-12),
    }


@torch.no_grad()
def validate_recon_4step(
    ema_model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    num_ode_steps: int = 4,
    method: str = "euler",
) -> dict[str, float]:
    """Cheap companion to ``validate_cfm``: 4-step Euler L1 over the full val set.

    Runs on the EMA model.  Complementary to ``val/velocity_mse``: catches
    cases where single-timestep velocity MSE looks fine but the integrated
    4-step output degrades.
    """
    ema_model.eval()
    sum_l1 = 0.0
    n_items = 0
    for batch in val_loader:
        x_0, x_1, f0, voicing, phoneme_ids, padding_mask = unpack_batch(batch, device)
        B = x_0.shape[0]

        pred = sample_ode(
            ema_model, x_0, f0, voicing, phoneme_ids,
            num_steps=num_ode_steps, method=method,
            padding_mask=padding_mask, diagnostics=False, cfg_scale=1.0,
        )
        sum_l1 += _masked_l1(pred, x_1, padding_mask) * B
        n_items += B

    n_items = max(1, n_items)
    return {"val/l1_reconstruction_4step": sum_l1 / n_items}
