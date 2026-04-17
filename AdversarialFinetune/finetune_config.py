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

    # -- CFG (applies only to the CFM phase; ODE phase always uses full cond)
    cfg_dropout_prob: float = 0.2
    sigma_min: float = 1.0e-4

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
    disc_channels: list = field(default_factory=lambda: [32, 64, 128, 256])
    crop_specs: list = field(default_factory=lambda: [
        [32, 64], [64, 128], [128, 128],
    ])

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
