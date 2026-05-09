"""VocaloFlow debugging script: DALI GT conditioning → inference → F0 comparison.

Loads a VocaloFlow model, uses DALI ground-truth notes as F0 conditioning
(with USTX-derived voicing, phonemes, and PL-BERT), runs inference, saves
all audio files, and plots F0 curves for comparison.
"""

import os
import shutil
import sys

import numpy as np
import soundfile as sf
import torch

# ── Path setup ──────────────────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))

if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from eval_utils.imports import setup_vocaloflow_sys_path, import_from_path

setup_vocaloflow_sys_path()

from inference.pipeline import (
    load_model,
    infer_chunked,
    mel_to_wav,
    parse_ustx,
    extract_prior_mel,
    get_note_coverage_voicing,
    build_phoneme_ids,
    build_plbert_frame_features,
)
from eval_utils.dali_ref import (
    load_dali_entry,
    get_chunk_time_range,
    extract_dali_f0_reference,
)
from eval_utils.audio import extract_f0

_data_helpers = import_from_path(
    "vf_data_helpers",
    os.path.join(REPO_ROOT, "VocaloFlow", "utils", "data_helpers.py"),
)
load_manifest = _data_helpers.load_manifest

# ── Configuration ───────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(
    REPO_ROOT, "VocaloFlow", "checkpoints", "4-27-wn", "checkpoint_150000.pt"
)
DATA_DIR = os.path.join(REPO_ROOT, "Data", "Rachie")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
DALI_ANNOT_DIR = os.path.join(REPO_ROOT, "DALI", "DALI_v2.0", "annot_tismir")
DALI_USTX_DIR = os.path.join(EVAL_DIR, "dali_ustx")
PHONESET_PATH = os.path.join(
    REPO_ROOT, "SoulX-Singer", "soulxsinger", "utils", "phoneme", "phone_set.json"
)
RMVPE_PATH = os.path.join(
    REPO_ROOT, "SoulX-Singer", "pretrained_models",
    "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt",
)
SPEAKER_EMB_PATH = os.path.join(
    REPO_ROOT, "SpeakerEmbedding", "embeddings", "Rachie", "speaker_embedding.pt"
)

NUM_ODE_STEPS = 16
ODE_METHOD = "midpoint"
CFG_SCALE = 1.0
CHUNK_SIZE = 256
OVERLAP = 16
SR = 24000
HOP = 480

OUTPUT_DIR = os.path.join(EVAL_DIR, "scratch_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Pick a sample ──────────────────────────────────────────────────────
print("Scanning for valid DALI USTX sample...")
dali_id = None
chunk_name = None
ustx_path = None
wav_path = None

for _did in sorted(os.listdir(DALI_USTX_DIR)):
    id_dir = os.path.join(DALI_USTX_DIR, _did)
    if not os.path.isdir(id_dir):
        continue
    dali_gz = os.path.join(DALI_ANNOT_DIR, f"{_did}.gz")
    if not os.path.exists(dali_gz):
        continue
    for _cn in sorted(os.listdir(id_dir)):
        chunk_dir = os.path.join(id_dir, _cn)
        _ustx = os.path.join(chunk_dir, "dali_prior.ustx")
        _wav = os.path.join(chunk_dir, "dali_prior.wav")
        if os.path.exists(_ustx) and os.path.exists(_wav):
            info = sf.info(_wav)
            if info.frames == 0:
                continue
            dali_id = _did
            chunk_name = _cn
            ustx_path = _ustx
            wav_path = _wav
            break
    if dali_id is not None:
        break

if dali_id is None:
    raise RuntimeError("No valid DALI USTX sample found")

print(f"Selected: {dali_id} / {chunk_name}")
print(f"  USTX: {ustx_path}")
print(f"  WAV:  {wav_path}")

# ── Load DALI annotation ───────────────────────────────────────────────
dali_entry = load_dali_entry(DALI_ANNOT_DIR, dali_id)
chunk_start_s, chunk_end_s = get_chunk_time_range(dali_entry, chunk_name)
print(f"Chunk time range: {chunk_start_s:.2f}s - {chunk_end_s:.2f}s "
      f"({chunk_end_s - chunk_start_s:.2f}s)")

# ── Parse USTX + extract prior mel ─────────────────────────────────────
notes_data = parse_ustx(ustx_path)
prior_mel = extract_prior_mel(wav_path)
T = prior_mel.shape[0]
print(f"Prior mel: {prior_mel.shape}, T={T} frames ({T * HOP / SR:.2f}s)")

# ── Find matching SoulX target in manifest ──────────────────────────────
print("Loading manifest to find SoulX target...")
manifest_df = load_manifest(MANIFEST_PATH, DATA_DIR)
match = manifest_df[
    (manifest_df["dali_id"] == dali_id) & (manifest_df["chunk_name"] == chunk_name)
]

soulx_target_wav_path = None
soulx_f0 = None
if len(match) > 0:
    row = match.iloc[0]
    sx_chunk_dir = os.path.dirname(row["phoneme_mask_path"])
    soulx_target_wav_path = os.path.join(sx_chunk_dir, "target.wav")
    soulx_f0_path = os.path.join(sx_chunk_dir, "target_f0.npy")

    if os.path.exists(soulx_f0_path):
        soulx_f0 = np.load(soulx_f0_path).astype(np.float32)
        print(f"SoulX F0: {soulx_f0.shape}")
    else:
        print(f"WARNING: target_f0.npy not found at {soulx_f0_path}")

    if os.path.exists(soulx_target_wav_path):
        print(f"SoulX target WAV: {soulx_target_wav_path}")
    else:
        print(f"WARNING: target.wav not found at {soulx_target_wav_path}")
        soulx_target_wav_path = None
else:
    print(f"WARNING: No manifest match for {dali_id}/{chunk_name}")

# ── Extract conditioning signals ────────────────────────────────────────
voicing = get_note_coverage_voicing(notes_data, T)
print(f"Voicing: {voicing.shape}, voiced={int(np.sum(voicing > 0))}/{T}")

dali_f0 = extract_dali_f0_reference(
    dali_entry, chunk_start_s, chunk_end_s, T, SR, HOP,
)
voiced_mask = dali_f0 > 0
print(f"DALI GT F0: {dali_f0.shape}, voiced={int(voiced_mask.sum())}/{T}")
if voiced_mask.any():
    print(f"  F0 range: {dali_f0[voiced_mask].min():.1f} - "
          f"{dali_f0[voiced_mask].max():.1f} Hz")

phoneme_ids = build_phoneme_ids(
    notes_data["notes"], notes_data["ms_per_tick"], T, PHONESET_PATH,
)
print(f"Phoneme IDs: {phoneme_ids.shape}, unique tokens={len(np.unique(phoneme_ids))}")

# ── Load model ──────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
model = load_model(CHECKPOINT_PATH, device)

# ── PL-BERT features (conditional) ─────────────────────────────────────
plbert_features = None
if model.config.use_plbert:
    print("Model uses PL-BERT - extracting features...")
    try:
        plbert_features = build_plbert_frame_features(
            notes_data["notes"], notes_data["ms_per_tick"], T,
            PHONESET_PATH, device=str(device),
        )
        print(f"PL-BERT features: {plbert_features.shape}")
    except (ModuleNotFoundError, ImportError) as e:
        print(f"WARNING: PL-BERT live extraction failed ({e}), "
              "falling back to precomputed features from SoulX chunk...")
        if len(match) > 0:
            from utils.resample import resolve_phoneme_indirection, resample_2d
            plbert_path = os.path.join(sx_chunk_dir, "plbert_features.npy")
            pmask_path = os.path.join(sx_chunk_dir, "phoneme_mask.npy")
            if os.path.exists(plbert_path) and os.path.exists(pmask_path):
                plbert_raw = np.load(plbert_path).astype(np.float32)
                pmask = np.load(pmask_path).astype(np.int64)
                mask_clipped = np.clip(pmask, 0, len(plbert_raw) - 1)
                plbert_frame = plbert_raw[mask_clipped]
                plbert_features = resample_2d(
                    torch.from_numpy(plbert_frame), T, mode="nearest",
                ).numpy().astype(np.float32)
                print(f"PL-BERT features (precomputed): {plbert_features.shape}")
            else:
                print("WARNING: precomputed PL-BERT files not found, skipping")

# ── Speaker embedding (conditional) ────────────────────────────────────
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

# ── VocaloFlow inference ────────────────────────────────────────────────
print("\nRunning VocaloFlow inference...")
with torch.no_grad():
    pred_mel = infer_chunked(
        model, prior_mel, dali_f0, voicing, phoneme_ids,
        chunk_size=CHUNK_SIZE, overlap=OVERLAP,
        num_steps=NUM_ODE_STEPS, method=ODE_METHOD,
        device=device, cfg_scale=CFG_SCALE,
        plbert_features=plbert_features,
        speaker_embedding=speaker_embedding,
    )
print(f"Predicted mel: {pred_mel.shape}")

print("Vocoding VF output...")
vf_audio = mel_to_wav(pred_mel)
print(f"VocaloFlow audio: {len(vf_audio)} samples ({len(vf_audio) / SR:.2f}s)")

# ── Save audio files ────────────────────────────────────────────────────
prefix = f"{dali_id}_{chunk_name}"

ustx_out = os.path.join(OUTPUT_DIR, f"{prefix}_ustx_prior.wav")
shutil.copy2(wav_path, ustx_out)
print(f"Saved USTX prior:    {ustx_out}")

if soulx_target_wav_path:
    soulx_out = os.path.join(OUTPUT_DIR, f"{prefix}_soulx_target.wav")
    shutil.copy2(soulx_target_wav_path, soulx_out)
    print(f"Saved SoulX target:  {soulx_out}")

vf_out = os.path.join(OUTPUT_DIR, f"{prefix}_vocaloflow.wav")
sf.write(vf_out, vf_audio, SR)
print(f"Saved VocaloFlow:    {vf_out}")

# ── Extract VocaloFlow F0 ──────────────────────────────────────────────
print("\nExtracting F0 from VocaloFlow output (RMVPE)...")
vf_f0 = extract_f0(vf_audio, RMVPE_PATH, str(device), SR, HOP)
print(f"VocaloFlow F0: {vf_f0.shape}")

# ── Plot F0 comparison ─────────────────────────────────────────────────
import matplotlib.pyplot as plt


def align_f0(f0_arr, target_len):
    if len(f0_arr) >= target_len:
        return f0_arr[:target_len]
    return np.pad(f0_arr, (0, target_len - len(f0_arr)))


def f0_for_plot(f0):
    out = f0.copy().astype(np.float64)
    out[out <= 0] = np.nan
    return out


t = np.arange(T) * HOP / SR

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(t, f0_for_plot(dali_f0), label="DALI GT F0",
        color="tab:green", linewidth=2, alpha=0.8)

if soulx_f0 is not None:
    sx_aligned = align_f0(soulx_f0, T)
    ax.plot(t, f0_for_plot(sx_aligned), label="SoulX target F0",
            color="tab:blue", linewidth=1.5, alpha=0.7)

vf_aligned = align_f0(vf_f0, T)
ax.plot(t, f0_for_plot(vf_aligned), label="VocaloFlow F0 (RMVPE)",
        color="tab:red", linewidth=1.5, alpha=0.7)

ax.set_xlabel("Time (s)")
ax.set_ylabel("F0 (Hz)")
ax.set_title(f"F0 Comparison: {dali_id} / {chunk_name}")
ax.legend(loc="upper right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_f0_comparison.png")
fig.savefig(plot_path, dpi=150)
print(f"\nSaved plot: {plot_path}")
plt.show()
