from dataclasses import dataclass


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
    input_channels: int = 385           # 128 (x_t) + 128 (prior) + 64 (f0) + 64 (ph) + 1 (vuv)
    dropout: float = 0.1

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

    # ── Inference ─────────────────────────────────────────────────────────
    num_ode_steps: int = 32
    ode_method: str = "midpoint"        # "euler" or "midpoint"

    # ── Paths ─────────────────────────────────────────────────────────────
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
