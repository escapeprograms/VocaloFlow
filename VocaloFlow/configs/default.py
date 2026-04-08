from dataclasses import dataclass


@dataclass
class VocaloFlowConfig:
    """Configuration for VocaloFlow conditional flow matching model."""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "../../Data/WillStetson"
    manifest_path: str = "../../Data/WillStetson/manifest.csv"
    mel_channels: int = 128
    sr: int = 24000
    hop: int = 480
    phoneme_vocab_size: int = 2820      # Full phone_set.json vocabulary
    phoneme_embed_dim: int = 256
    max_seq_len: int = 256              # Crop/pad to this (covers p95)
    max_dtw_cost: float = 200.0         # Quality filter threshold
    val_fraction: float = 0.05
    seed: int = 42

    # ── Model ─────────────────────────────────────────────────────────────
    hidden_dim: int = 1024
    num_heads: int = 16
    head_dim: int = 64                  # hidden_dim // num_heads
    ffn_dim: int = 4096
    num_dit_blocks: int = 2
    input_channels: int = 514           # 128 + 128 + 1 + 1 + 256

    # ── Training ──────────────────────────────────────────────────────────
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 200_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    log_every: int = 50
    val_every: int = 2000
    save_every: int = 5000

    # ── Flow Matching ─────────────────────────────────────────────────────
    sigma_min: float = 1e-4

    # ── Inference ─────────────────────────────────────────────────────────
    num_ode_steps: int = 32
    ode_method: str = "midpoint"        # "euler" or "midpoint"

    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
