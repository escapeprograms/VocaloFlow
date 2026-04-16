from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class VocaloFlowConfig:
    """Configuration for VocaloFlow conditional flow matching model."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "../Data/Rachie"
    manifest_path: str = "../Data/Rachie/manifest.csv"
    mel_channels: int = 128
    sr: int = 24000
    hop: int = 480
    phoneme_vocab_size: int = 2820      # Full phone_set.json vocabulary
    phoneme_embed_dim: int = 64
    f0_embed_dim: int = 64
    max_seq_len: int = 256              # Crop/pad to this (covers p95)
    max_dtw_cost: float = 100.0         # Quality filter threshold
    val_fraction: float = 0.05
    seed: int = 42
    split_mode: str = "song"            # "song" (split_by_song) or "random" (split_random)

    # ── Model ─────────────────────────────────────────────────────────────
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 64                  # hidden_dim // num_heads
    ffn_dim: int = 2048
    num_dit_blocks: int = 6
    num_convnext_blocks: int = 4        # ConvNeXtV2 pre-processing blocks (0 = disable)
    convnext_kernel_size: int = 7       # Depthwise conv kernel (7 @ 20ms/frame = 140ms)
    input_channels: int = 385           # 128 (x_t) + 128 (prior) + 64 (f0) + 64 (ph) + 1 (vuv)
    dropout: float = 0.1

    # ── Phoneme Blurring ──────────────────────────────────────────────────
    phoneme_blur_enabled: bool = False   # Use BlurredPhonemeEmbedding vs hard lookup
    phoneme_blend_fraction: float = 0.2 # Blend radius as fraction of shorter phoneme duration

    # ── WaveNet Pre-processing ────────────────────────────────────────────
    # Optional alternative to ConvNeXt. Dilated conv + gated activation with
    # per-layer timestep conditioning (DiffSinger/PWG style). 0 = disabled.
    num_wavenet_blocks: int = 0
    wavenet_kernel_size: int = 3
    wavenet_dilation_cycle: int = 4      # dilations: 1, 2, 4, 8 repeating
    wavenet_skip_channels: int = 512     # drop to 256 to save params if needed
    wavenet_dropout: float = 0.1

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 200_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 1000
    save_every: int = 5000

    # ── Classifier-Free Guidance ──────────────────────────────────────────
    cfg_dropout_prob: float = 0.2       # Drop conditioning this fraction of time
    cfg_scale: float = 2.0              # Guidance scale at inference

    # ── Flow Matching ─────────────────────────────────────────────────────
    sigma_min: float = 1e-4

    # ── STFT Auxiliary Loss ───────────────────────────────────────────────
    stft_loss_enabled: bool = False
    stft_loss_lambda: float = 0.1
    # (n_fft, hop, win) per resolution, operating on the mel time axis.
    # List-of-lists (not tuples) so YAML safe_load round-trips cleanly.
    stft_resolutions: list = field(default_factory=lambda: [
        [16, 4, 16],
        [32, 8, 32],
        [64, 16, 64],
    ])

    # ── Inference ─────────────────────────────────────────────────────────
    num_ode_steps: int = 32
    ode_method: str = "midpoint"        # "euler" or "midpoint"

    # ── Paths ─────────────────────────────────────────────────────────────
    run_name: str = ""                  # Set via --name; determines checkpoint/config/log subdirs
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"

    # ── YAML serialization ────────────────────────────────────────────────

    def to_yaml(self, path: str) -> None:
        """Save config as a YAML file."""
        import yaml

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str) -> "VocaloFlowConfig":
        """Load config from a YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
