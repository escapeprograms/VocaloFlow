"""Configuration dataclass for adversarial flow-matching fine-tuning of VocaloFlow."""

from dataclasses import asdict, dataclass, field


@dataclass
class FinetuneConfig:
    """Configuration for adversarial FM fine-tuning of a pretrained VocaloFlow.

    Everything toggleable is exposed as a boolean so individual loss components
    can be validated in isolation during smoke tests.
    """

    # -- Pretrained source ------------------------------------------------
    pretrained_run: str = "4-16-wavenet"          # subdirectory of VocaloFlow/checkpoints/
    pretrained_step: int = 0                      # 0 -> use latest; else exact checkpoint_<step>.pt
    pretrained_vocaloflow_dir: str = "../VocaloFlow"

    # -- Data (mirror VocaloFlow defaults) --------------------------------
    data_dir: str = "../Data/Rachie"
    manifest_path: str = "../Data/Rachie/manifest.csv"
    max_seq_len: int = 256
    max_dtw_cost: float = 100.0
    val_fraction: float = 0.05
    split_mode: str = "song"                      # "song" or "random"
    seed: int = 42

    # -- ODE unrolling (training-time) ------------------------------------
    ode_num_steps: int = 4
    ode_method: str = "euler"                     # "euler" or "midpoint"
    ode_grad_checkpoint: bool = False             # torch.utils.checkpoint per step to save VRAM

    # -- Loss toggles -----------------------------------------------------
    enable_cfm: bool = True                       # velocity-matching anchor
    enable_reconstruction: bool = True            # L1(x_1_hat, target_mel)
    enable_adversarial: bool = True               # hinge GAN
    enable_feature_matching: bool = True          # discriminator feature L1

    # -- Loss weights -----------------------------------------------------
    lambda_cfm: float = 1.0
    lambda_rec: float = 1.0
    lambda_adv: float = 0.1
    lambda_fm: float = 2.0
    lambda_cfm_final: float = 1.0                # end-of-training value (1.0 = no decay)

    # -- CFG (applies only to the CFM phase; ODE phase always uses full cond)
    cfg_dropout_prob: float = 0.2
    sigma_min: float = 1.0e-4

    # -- Energy-Balanced Loss (CFM anchor) --------------------------------
    energy_balanced_loss: bool = False
    energy_balance_lambda: float = 0.4
    energy_balance_epsilon: float = 1e-4
    energy_balance_mode: str = "both"

    # -- Warmup schedule --------------------------------------------------
    disc_warmup_steps: int = 3000                 # steps from fine-tune start with lambda_adv = lambda_fm = 0
    adv_ramp_steps: int = 2000                    # linear ramp width after warmup

    # -- Optimizers -------------------------------------------------------
    gen_learning_rate: float = 2.0e-5             # constant (no scheduler) — see spec 7.2
    disc_learning_rate: float = 2.0e-4
    disc_lr_warmup_steps: int = 1000              # linear warmup for disc LR (then cosine)
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    reset_gen_optimizer: bool = True              # drop pretrained Adam moments (LR changed 5x)

    # -- EMA --------------------------------------------------------------
    ema_decay: float = 0.9999
    preserve_global_step: bool = True             # continue step counter from pretrained ckpt

    # -- Training loop ----------------------------------------------------
    batch_size: int = 8                           # reduced from pretraining (ODE unroll ~4x mem)
    total_finetune_steps: int = 30000             # added on top of starting global_step
    log_every: int = 50
    val_every: int = 1000
    eval_every: int = 3000                        # true-inference eval (32-step midpoint)
    save_every: int = 5000
    eval_num_samples: int = 8                     # val subset size for heavy eval
    save_mel_plots: bool = True                   # save 4-panel mel PNGs every eval_every steps
    num_mel_plots: int = 4                        # cap on PNGs saved (and uploaded to wandb) per eval

    # -- Discriminator ---------------------------------------------------
    disc_type: str = "patch"                          # "patch" (PatchDiscriminator) or "dit" (DiTDiscriminator)
    disc_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    crop_specs: list = field(default_factory=lambda: [
        [32, 64], [64, 128], [128, 128],
    ])
    disc_dit_num_blocks: int = 4
    disc_dit_hidden_dim: int = 512
    disc_dit_num_heads: int = 8
    disc_dit_ffn_dim: int = 2048
    disc_dit_feature_blocks: list = field(default_factory=lambda: [1, 3])
    disc_steps_per_gen: int = 1                       # D updates per G update (>1 uses fresh batches)
    logit_center_weight: float = 0.0                  # 0 = disabled; e.g. 0.01 to penalize logit drift

    # -- Gradient normalization (Exp 4) ---------------------------------
    enable_grad_norm: bool = False                    # separate adv backward + L2-normalize

    # -- Discriminator augmentation (Exp 4) -----------------------------
    enable_disc_augmentation: bool = False
    disc_aug_prob: float = 0.5                        # per-augmentation fire probability
    disc_aug_max_shift: int = 16                      # circular temporal shift range (frames)
    disc_aug_cutout_min: int = 8                      # temporal cutout minimum width (frames)
    disc_aug_cutout_max: int = 32                     # temporal cutout maximum width (frames)
    enable_freq_cutout: bool = False                  # optional frequency-band cutout
    disc_aug_freq_cutout_min: int = 8
    disc_aug_freq_cutout_max: int = 32

    # -- High-frequency auxiliary discriminator ----------------------------
    use_hf_discriminator: bool = False                   # master toggle; False = zero code paths change
    hf_disc_hf_start: int = 64                           # first mel bin of HF band (inclusive)
    hf_disc_num_blocks: int = 2
    hf_disc_hidden_dim: int = 512
    hf_disc_num_heads: int = 8
    hf_disc_ffn_dim: int = 2048
    hf_disc_feature_blocks: list = field(default_factory=lambda: [1])
    lambda_hf_adv: float = 1.0                           # base weight (gradient norm bounds magnitude)
    hf_disc_learning_rate: float = 2.0e-4
    hf_disc_lr_warmup_steps: int = 1000

    # -- Paths ------------------------------------------------------------
    run_name: str = ""
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # -- YAML serialization ----------------------------------------------

    def to_yaml(self, path: str) -> None:
        """Save config as a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "FinetuneConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
