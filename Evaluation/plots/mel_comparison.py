"""Mel spectrogram comparison: 4-panel figure for a quality-context sample.

Side-by-side comparison of mel spectrograms from:
  1. Prior (OpenUtau synthesis)
  2. SoulX Target (ground truth singing)
  3. VocaloFlow (4-27-wn checkpoint)
  4. RVC Baseline (pre-generated voice conversion)

Usage (from the Evaluation/ folder):
    python -m plots.mel_comparison
"""

import os
import sys

import librosa
import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))

if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from eval_utils.imports import setup_vocaloflow_sys_path

setup_vocaloflow_sys_path()

from inference.pipeline import load_model, infer_chunked
from eval_utils.audio import audio_to_mel
from eval_utils.val_set import get_val_manifest
from utils.resample import resolve_phoneme_indirection, resample_2d

# ── Configuration ───────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(
    REPO_ROOT, "VocaloFlow", "checkpoints", "4-27-wn", "checkpoint_150000.pt"
)
DATA_DIR = os.path.join(REPO_ROOT, "Data", "Rachie")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
RVC_OUTPUT_DIR = os.path.join(REPO_ROOT, "RVCBaseline", "output")
SPEAKER_EMB_PATH = os.path.join(
    REPO_ROOT, "SpeakerEmbedding", "embeddings", "Rachie", "speaker_embedding.pt"
)

MAX_DTW_COST = 100.0
VAL_FRACTION = 0.05
SEED = 42

NUM_ODE_STEPS = 16
ODE_METHOD = "midpoint"
CFG_SCALE = 1.0
CHUNK_SIZE = 256
OVERLAP = 16
SR = 24000
HOP = 480

SAMPLE_SEED = 12

OUTPUT_DIR = os.path.join(_THIS_DIR, "mel_comparison_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ─────────────────────────────────────────────────────────────

def load_chunk_inputs(row, use_plbert):
    """Load all model inputs for a single validation chunk."""
    prior_mel = np.load(row["prior_mel_path"]).astype(np.float32)
    target_mel = np.load(row["target_mel_path"]).astype(np.float32)
    f0 = np.load(row["f0_path"]).astype(np.float32)
    voicing = np.load(row["voicing_path"]).astype(np.float32)

    chunk_dir = os.path.dirname(row["prior_mel_path"])
    phoneme_mask = np.load(row["phoneme_mask_path"]).astype(np.int64)
    phoneme_ids_raw = np.load(
        os.path.join(chunk_dir, "phoneme_ids.npy")
    ).astype(np.int64)
    resolved = resolve_phoneme_indirection(phoneme_ids_raw, phoneme_mask)

    T = target_mel.shape[0]

    def match_1d(x, n):
        return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))

    def match_2d(x, n):
        return x[:n] if x.shape[0] >= n else np.pad(x, ((0, n - x.shape[0]), (0, 0)))

    prior_mel = match_2d(prior_mel, T)
    f0 = match_1d(f0, T)
    voicing = match_1d(voicing, T)
    resolved = match_1d(resolved, T)

    result = {
        "prior_mel": prior_mel,
        "target_mel": target_mel,
        "f0": f0,
        "voicing": voicing,
        "phoneme_ids": resolved,
        "chunk_dir": chunk_dir,
    }

    if use_plbert:
        plbert_path = os.path.join(chunk_dir, "plbert_features.npy")
        if os.path.exists(plbert_path):
            plbert_feats = np.load(plbert_path).astype(np.float32)
            mask_clipped = np.clip(phoneme_mask, 0, len(plbert_feats) - 1)
            plbert_frame = plbert_feats[mask_clipped]
            plbert_frame = match_2d(plbert_frame, T)
            result["plbert_features"] = plbert_frame
        else:
            print(f"  WARNING: plbert_features.npy not found in {chunk_dir}")
            result["plbert_features"] = None

    return result


# ── Sample selection ────────────────────────────────────────────────────
print("Loading validation manifest...")
val_df = get_val_manifest(DATA_DIR, MANIFEST_PATH, MAX_DTW_COST, VAL_FRACTION, SEED)

print("Finding samples with matching RVC output...")
valid_indices = []
for i in range(len(val_df)):
    row = val_df.iloc[i]
    rvc_path = os.path.join(
        RVC_OUTPUT_DIR, f"{row['dali_id']}_{row['chunk_name']}.wav"
    )
    if os.path.exists(rvc_path):
        valid_indices.append(i)

if not valid_indices:
    raise RuntimeError("No validation samples have matching RVC output")

print(f"Found {len(valid_indices)} samples with RVC output.")

durations = []
for vi in valid_indices:
    r = val_df.iloc[vi]
    t_mel = np.load(r["target_mel_path"]).shape[0]
    durations.append(t_mel)

ranked = sorted(zip(valid_indices, durations), key=lambda x: -x[1])
row = val_df.iloc[ranked[0][0]]
print(f"Longest sample: T={ranked[0][1]} frames ({ranked[0][1] * HOP / SR:.2f}s)")

dali_id = row["dali_id"]
chunk_name = row["chunk_name"]
rvc_wav_path = os.path.join(RVC_OUTPUT_DIR, f"{dali_id}_{chunk_name}.wav")

print(f"Selected: {dali_id} / {chunk_name}")

# ── Load model ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
model = load_model(CHECKPOINT_PATH, device)

speaker_embedding = None
if model.config.use_speaker_embedding:
    if os.path.exists(SPEAKER_EMB_PATH):
        speaker_embedding = (
            torch.load(SPEAKER_EMB_PATH, weights_only=True)
            .float().unsqueeze(0).to(device)
        )
        print(f"Speaker embedding: {speaker_embedding.shape}")
    else:
        print(f"WARNING: speaker embedding not found at {SPEAKER_EMB_PATH}")

# ── Load data ───────────────────────────────────────────────────────────
print("Loading chunk inputs...")
inputs = load_chunk_inputs(row, model.config.use_plbert)

print("Loading RVC audio and converting to mel...")
rvc_audio, _ = librosa.load(rvc_wav_path, sr=SR)
rvc_mel = audio_to_mel(rvc_audio, SR)
print(f"RVC mel: {rvc_mel.shape}")

# ── VocaloFlow inference ────────────────────────────────────────────────
print("Running VocaloFlow inference...")
with torch.no_grad():
    pred_mel = infer_chunked(
        model, inputs["prior_mel"], inputs["f0"], inputs["voicing"],
        inputs["phoneme_ids"],
        chunk_size=CHUNK_SIZE, overlap=OVERLAP,
        num_steps=NUM_ODE_STEPS, method=ODE_METHOD,
        device=device, cfg_scale=CFG_SCALE,
        plbert_features=inputs.get("plbert_features"),
        speaker_embedding=speaker_embedding,
    )
print(f"Predicted mel: {pred_mel.shape}")

# ── Align lengths ───────────────────────────────────────────────────────
all_mels_raw = [inputs["prior_mel"], inputs["target_mel"], pred_mel, rvc_mel]
min_T = min(m.shape[0] for m in all_mels_raw)
all_mels = [m[:min_T] for m in all_mels_raw]
all_titles = ["Prior (OpenUtau)", "SoulX Target", "VocaloFlow", "RVC Baseline"]

print(f"Aligned all mels to T={min_T} frames ({min_T * HOP / SR:.2f}s)")

# ── Plot ────────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

vmin = min(m.min() for m in all_mels)
vmax = max(m.max() for m in all_mels)
t_end = min_T * HOP / SR

fig, axes = plt.subplots(1, 4, figsize=(24, 5))

for i, (ax, mel, title) in enumerate(zip(axes, all_mels, all_titles)):
    im = ax.imshow(
        mel.T, origin="lower", aspect="auto", cmap="magma",
        vmin=vmin, vmax=vmax,
        extent=[0, t_end, 0, 128],
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=12)
    if i == 0:
        ax.set_ylabel("Mel bin", fontsize=12)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=10)

plt.tight_layout()
fig.subplots_adjust(right=0.92)
cbar_ax = fig.add_axes([0.93, 0.12, 0.01, 0.76])
fig.colorbar(im, cax=cbar_ax, label="Normalized amplitude")

prefix = f"{dali_id}_{chunk_name}"
plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_mel_comparison.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved plot: {plot_path}")
plt.show()

# ── Delta-mel plots (difference vs SoulX Target) ──────────────────────
target_mel = all_mels[1]
vf_diff = all_mels[2] - target_mel
rvc_diff = all_mels[3] - target_mel

vlim = max(np.abs(vf_diff).max(), np.abs(rvc_diff).max())

diff_mels = [vf_diff, rvc_diff]
diff_titles = [
    f"VocaloFlow − SoulX Target",
    f"RVC Baseline − SoulX Target",
]

print(f"\n[VocaloFlow]   mean |Δ| = {np.abs(vf_diff).mean():.4f},  max |Δ| = {np.abs(vf_diff).max():.4f}")
print(f"[RVC Baseline] mean |Δ| = {np.abs(rvc_diff).mean():.4f},  max |Δ| = {np.abs(rvc_diff).max():.4f}")

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))

for i, (ax, diff, title) in enumerate(zip(axes2, diff_mels, diff_titles)):
    im2 = ax.imshow(
        diff.T, origin="lower", aspect="auto", cmap="coolwarm",
        vmin=-vlim, vmax=vlim,
        extent=[0, t_end, 0, 128],
    )
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Time (s)", fontsize=12)
    if i == 0:
        ax.set_ylabel("Mel bin", fontsize=12)
    else:
        ax.set_ylabel("")
    ax.tick_params(labelsize=10)

plt.tight_layout()
fig2.subplots_adjust(right=0.92)
cbar_ax2 = fig2.add_axes([0.93, 0.12, 0.01, 0.76])
fig2.colorbar(im2, cax=cbar_ax2, label="Δ amplitude")

delta_path = os.path.join(OUTPUT_DIR, f"{prefix}_mel_delta.png")
fig2.savefig(delta_path, dpi=300, bbox_inches="tight")
print(f"Saved delta plot: {delta_path}")
plt.show()
