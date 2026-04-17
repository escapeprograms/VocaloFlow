"""Configuration dataclass for the AdversarialPostnet experiment."""

from dataclasses import asdict, dataclass, field


@dataclass
class PostnetConfig:
    """Configuration for the adversarial post-network experiment.

    Covers both the prediction generation phase (running frozen VocaloFlow)
    and the subsequent postnet training phase.
    """

    # -- Data (generation) -------------------------------------------------
    data_dir: str = "../Data/Rachie"
    manifest_path: str = "../Data/Rachie/manifest.csv"
    vocaloflow_checkpoint: str = (
        "../VocaloFlow/checkpoints/4-16-wavenet/checkpoint_55000.pt"
    )
    max_dtw_cost: float = 100.0          # Quality filter (match VocaloFlow)
    max_songs: int = 0                   # 0 = all songs; >0 = subset
    song_subset_seed: int = 42           # Seed for reproducible song subsetting

    # -- Data (training) ---------------------------------------------------
    predicted_mel_manifest: str = ""     # Output of generate_predictions
    max_seq_len: int = 256
    val_fraction: float = 0.05           # Match VocaloFlow default
    seed: int = 42                       # Same seed => same song-level split
    split_mode: str = "song"             # "song" or "random"

    # -- Inference (for generation) ----------------------------------------
    num_ode_steps: int = 32
    ode_method: str = "midpoint"
    cfg_scale: float = 2.0
    generation_batch_size: int = 16

    # -- Training -----------------------------------------------------------
    batch_size: int = 32

    # -- PostNet architecture -----------------------------------------------
    postnet_num_blocks: int = 4
    postnet_kernel_size: int = 3

    # -- Discriminator architecture -----------------------------------------
    disc_channels: list = field(default_factory=lambda: [32, 64, 128, 256])

    # -- Loss weights -------------------------------------------------------
    lambda_adv: float = 0.1
    lambda_fm: float = 2.0

    # -- Training hyperparameters -------------------------------------------
    learning_rate: float = 2e-4
    disc_learning_rate: float = 2e-4
    adam_beta1: float = 0.8
    adam_beta2: float = 0.99
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    total_steps: int = 20000
    warmup_steps: int = 0              # LR warmup steps
    disc_warmup_steps: int = 2000      # Steps before adversarial loss activates
    adv_ramp_steps: int = 1000         # Steps to ramp lambda_adv from 0 to target

    # -- Logging & checkpointing --------------------------------------------
    log_every: int = 50
    val_every: int = 1000
    save_every: int = 5000

    # -- Discriminator crop specs: [time_frames, mel_bins] per scale --------
    crop_specs: list = field(default_factory=lambda: [
        [32, 64], [64, 128], [128, 128],
    ])

    # -- Paths -------------------------------------------------------------
    run_name: str = ""
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # -- YAML serialization ------------------------------------------------

    def to_yaml(self, path: str) -> None:
        """Save config as a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "PostnetConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
