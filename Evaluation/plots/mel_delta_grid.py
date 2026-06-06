"""Mel spectrogram delta grid: 13-panel figure comparing all experiments.

Shows (predicted − SoulX Target) mel-spectrogram deltas for every experiment
in the paper's Table 1, grouped by development phase.

Usage (from the Evaluation/ folder):
    python -m plots.mel_delta_grid
"""

import os
import sys

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))

if EVAL_DIR not in sys.path:
    sys.path.insert(0, EVAL_DIR)

from eval_utils.imports import setup_vocaloflow_sys_path, import_from_path

setup_vocaloflow_sys_path()

from inference.pipeline import load_model, infer_chunked
from eval_utils.val_set import get_val_manifest
from utils.resample import resolve_phoneme_indirection

# ── Experiment configuration ────────────────────────────────────────────

EXPERIMENTS = [
    # Row 0: Base Model Development
    dict(label="v1", desc="6 DiT, discrete ph.",
         checkpoint="VocaloFlow/checkpoints/zany-armadillo-6/checkpoint_10000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=0),
    dict(label="v2", desc="+ConvNeXtV2, +STFT",
         checkpoint="VocaloFlow/checkpoints/4-13-stft/checkpoint_45000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=0),
    dict(label="v3", desc="8 WN + 6 DiT",
         checkpoint="VocaloFlow/checkpoints/4-16-wavenet/checkpoint_55000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=0),
    # Row 1: Adversarial Fine-Tuning
    dict(label="Exp 1", desc="PostNet (frozen)",
         checkpoint="AdversarialPostnet/checkpoints/my-run/checkpoint_20000.pt",
         type="postnet",
         base_checkpoint="VocaloFlow/checkpoints/4-16-wavenet/checkpoint_55000.pt",
         ode_steps=32, ode_method="midpoint", row=1),
    dict(label="Exp 2", desc="AFT + L1 recon",
         checkpoint="AdversarialFinetune/checkpoints/4-17-adv/checkpoint_65000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=1),
    dict(label="Exp 3", desc="Drop L1, scale D",
         checkpoint="AdversarialFinetune/checkpoints/4-17-no-rec/checkpoint_60000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=1),
    dict(label="Exp 4", desc="Full AF recipe",
         checkpoint="AdversarialFinetune/checkpoints/4-18-afm2/checkpoint_75000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=1),
    # Row 2: Conditioning & Loss
    dict(label="Exp 5", desc="PL-BERT + AFT",
         checkpoint="AdversarialFinetune/checkpoints/4-19-pl-bert-ft/checkpoint_75000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=2),
    dict(label="Exp 6", desc="Energy-balanced",
         checkpoint="AdversarialFinetune/checkpoints/4-20-energy-ft/checkpoint_75000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=2),
    # Row 3: Scaling & Architecture
    dict(label="Exp 7", desc="Data 25→80 hrs",
         checkpoint="VocaloFlow/checkpoints/4-23-embed/checkpoint_115000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=3),
    dict(label="Exp 8", desc="+Spk emb, PL fix",
         checkpoint="AdversarialFinetune/checkpoints/4-23-embed-ft/checkpoint_85000.pt",
         type="adversarial_ft", ode_steps=4, ode_method="euler", row=3),
    dict(label="Exp 9", desc="12 WN+10 DiT, 85M",
         checkpoint="VocaloFlow/checkpoints/4-25-big/checkpoint_175000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=3),
    dict(label="Exp 10", desc="Pure WN 20L",
         checkpoint="VocaloFlow/checkpoints/4-26-diffsinger/checkpoint_150000.pt",
         type="vocaloflow", ode_steps=16, ode_method="midpoint", row=3),
]

# ── Shared constants ────────────────────────────────────────────────────
DATA_DIR = os.path.join(REPO_ROOT, "Data", "Rachie")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")
SPEAKER_EMB_PATH = os.path.join(
    REPO_ROOT, "SpeakerEmbedding", "embeddings", "Rachie", "speaker_embedding.pt"
)

MAX_DTW_COST = 100.0
VAL_FRACTION = 0.05
SEED = 42
CHUNK_SIZE = 256
OVERLAP = 16
CFG_SCALE = 1.0
SR = 24000
HOP = 480

OUTPUT_DIR = os.path.join(_THIS_DIR, "mel_delta_grid_output")
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
            result["plbert_features"] = None

    return result


def load_speaker_embedding(model, device):
    """Load speaker embedding if the model config requires it."""
    if not getattr(model.config, "use_speaker_embedding", False):
        return None
    if not os.path.exists(SPEAKER_EMB_PATH):
        print(f"  WARNING: speaker embedding not found at {SPEAKER_EMB_PATH}")
        return None
    emb = (
        torch.load(SPEAKER_EMB_PATH, weights_only=True)
        .float().unsqueeze(0).to(device)
    )
    return emb


def run_vocaloflow_inference(exp, inputs, device):
    """Standard VocaloFlow ODE inference."""
    ckpt_path = os.path.join(REPO_ROOT, exp["checkpoint"])
    print(f"  Loading {exp['checkpoint']}...")
    model = load_model(ckpt_path, device)
    speaker_emb = load_speaker_embedding(model, device)

    plbert = inputs.get("plbert_features")
    if not getattr(model.config, "use_plbert", False):
        plbert = None

    with torch.no_grad():
        pred_mel = infer_chunked(
            model, inputs["prior_mel"], inputs["f0"], inputs["voicing"],
            inputs["phoneme_ids"],
            chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            num_steps=exp["ode_steps"], method=exp["ode_method"],
            device=device, cfg_scale=CFG_SCALE,
            plbert_features=plbert,
            speaker_embedding=speaker_emb,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_mel


def run_postnet_inference(exp, inputs, device):
    """Two-stage: VocaloFlow base → PostNet refinement."""
    base_ckpt = os.path.join(REPO_ROOT, exp["base_checkpoint"])
    postnet_ckpt_path = os.path.join(REPO_ROOT, exp["checkpoint"])

    print(f"  Loading base model {exp['base_checkpoint']}...")
    base_model = load_model(base_ckpt, device)
    speaker_emb = load_speaker_embedding(base_model, device)

    plbert = inputs.get("plbert_features")
    if not getattr(base_model.config, "use_plbert", False):
        plbert = None

    with torch.no_grad():
        base_mel = infer_chunked(
            base_model, inputs["prior_mel"], inputs["f0"], inputs["voicing"],
            inputs["phoneme_ids"],
            chunk_size=CHUNK_SIZE, overlap=OVERLAP,
            num_steps=exp["ode_steps"], method=exp["ode_method"],
            device=device, cfg_scale=CFG_SCALE,
            plbert_features=plbert,
            speaker_embedding=speaker_emb,
        )

    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    PostNet = import_from_path(
        "ap_postnet",
        os.path.join(REPO_ROOT, "AdversarialPostnet", "model", "postnet.py"),
    ).PostNet
    import_from_path(
        "configs.postnet_config",
        os.path.join(REPO_ROOT, "AdversarialPostnet", "configs", "postnet_config.py"),
    )

    print(f"  Loading PostNet {exp['checkpoint']}...")
    ckpt = torch.load(postnet_ckpt_path, map_location=device, weights_only=False)
    postnet = PostNet().to(device)
    postnet.load_state_dict(ckpt["postnet_state_dict"])
    postnet.eval()

    with torch.no_grad():
        mel_t = torch.from_numpy(base_mel).unsqueeze(0).float().to(device)
        pred_mel = postnet(mel_t).squeeze(0).cpu().numpy()

    del postnet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred_mel


# ── Sample selection ────────────────────────────────────────────────────
FORCE_DALI_ID = "c68c53e45597455e90262e3de3469d87"
FORCE_CHUNK = "line_4"

print("Loading validation manifest...")
val_df = get_val_manifest(DATA_DIR, MANIFEST_PATH, MAX_DTW_COST, VAL_FRACTION, SEED)

if FORCE_DALI_ID and FORCE_CHUNK:
    match = val_df[
        (val_df["dali_id"] == FORCE_DALI_ID) & (val_df["chunk_name"] == FORCE_CHUNK)
    ]
    if match.empty:
        raise RuntimeError(
            f"Sample {FORCE_DALI_ID}/{FORCE_CHUNK} not found in val manifest"
        )
    row = match.iloc[0]
else:
    durations = []
    for i in range(len(val_df)):
        r = val_df.iloc[i]
        t_mel = np.load(r["target_mel_path"]).shape[0]
        durations.append((i, t_mel))
    ranked = sorted(durations, key=lambda x: -x[1])
    row = val_df.iloc[ranked[0][0]]

dali_id = row["dali_id"]
chunk_name = row["chunk_name"]
T_target = np.load(row["target_mel_path"]).shape[0]
print(f"Selected: {dali_id} / {chunk_name}  "
      f"(T={T_target} frames, {T_target * HOP / SR:.2f}s)")

# ── Load inputs ─────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

inputs = load_chunk_inputs(row, use_plbert=True)
target_mel = inputs["target_mel"]

# ── Inference loop (with disk cache) ────────────────────────────────────
CACHE_DIR = os.path.join(OUTPUT_DIR, "_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

delta_mels = {}
stats = {}

for exp in EXPERIMENTS:
    cache_key = exp["label"].replace(" ", "_")
    cache_path = os.path.join(CACHE_DIR, f"{dali_id}_{chunk_name}_{cache_key}.npy")

    if os.path.exists(cache_path):
        print(f"[{exp['label']}] cached")
        delta = np.load(cache_path)
    else:
        print(f"\n[{exp['label']}] {exp['desc']}")
        if exp["type"] == "postnet":
            pred_mel = run_postnet_inference(exp, inputs, device)
        else:
            pred_mel = run_vocaloflow_inference(exp, inputs, device)

        T_min = min(pred_mel.shape[0], target_mel.shape[0])
        delta = pred_mel[:T_min] - target_mel[:T_min]
        np.save(cache_path, delta)

    delta_mels[exp["label"]] = delta

    mean_abs = np.abs(delta).mean()
    max_abs = np.abs(delta).max()
    stats[exp["label"]] = (mean_abs, max_abs)
    print(f"  mean |Δ| = {mean_abs:.4f},  max |Δ| = {max_abs:.4f}")

# ── Figure ──────────────────────────────────────────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

all_abs = np.concatenate([np.abs(d).ravel() for d in delta_mels.values()])
vlim = float(np.percentile(all_abs, 99.5))

ROW_CFG = [
    {"title": "Base Model Development",     "n": 3},
    {"title": "Adversarial Fine-Tuning",     "n": 4},
    {"title": "Conditioning & Loss",         "n": 2},
    {"title": "Scaling & Architecture",      "n": 4},
]

fig = plt.figure(figsize=(7.5, 10.0))

outer_gs = gridspec.GridSpec(
    4, 1, figure=fig,
    hspace=0.85,
    top=0.97, bottom=0.03, left=0.07, right=0.89,
)

exp_iter = iter(EXPERIMENTS)
all_ims = []

for row_idx, rcfg in enumerate(ROW_CFG):
    n = rcfg["n"]
    if n == 2:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer_gs[row_idx], wspace=0.25
        )
        col_positions = [(1, 2), (2, 3)]
    elif n == 3:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer_gs[row_idx], wspace=0.25
        )
        col_positions = [(0, 1), (1, 2), (2, 3)]
    else:
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer_gs[row_idx], wspace=0.25
        )
        col_positions = [(0, 1), (1, 2), (2, 3), (3, 4)]

    for panel_idx, (c0, c1) in enumerate(col_positions):
        exp = next(exp_iter)
        ax = fig.add_subplot(inner_gs[0, c0:c1])

        delta = delta_mels[exp["label"]]
        t_end = delta.shape[0] * HOP / SR

        im = ax.imshow(
            delta.T, origin="lower", aspect="auto",
            cmap="coolwarm", vmin=-vlim, vmax=vlim,
            extent=[0, t_end, 0, 128],
        )
        all_ims.append(im)

        mean_abs = stats[exp["label"]][0]
        ax.set_title(
            f"{exp['label']}: {exp['desc']}",
            fontsize=6.5, fontweight="bold", pad=3,
        )
        ax.text(
            0.98, 0.04, f"|Δ|={mean_abs:.3f}",
            transform=ax.transAxes, fontsize=5,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.8, lw=0.3),
        )
        ax.tick_params(labelsize=5)
        ax.set_xlabel("Time (s)", fontsize=5.5)
        if panel_idx == 0:
            ax.set_ylabel("Mel bin", fontsize=5.5)
        else:
            ax.set_ylabel("")

    bbox = outer_gs[row_idx].get_position(fig)
    fig.text(
        0.48, bbox.y1 + 0.018,
        rcfg["title"],
        ha="center", fontsize=7, fontstyle="italic", color="0.35",
    )

cbar_ax = fig.add_axes([0.91, 0.04, 0.015, 0.92])
fig.colorbar(all_ims[0], cax=cbar_ax)
cbar_ax.set_ylabel("Δ amplitude", fontsize=6.5)
cbar_ax.tick_params(labelsize=5)

prefix = f"{dali_id}_{chunk_name}"
plot_path = os.path.join(OUTPUT_DIR, f"{prefix}_mel_delta_grid.png")
fig.savefig(plot_path, dpi=300, bbox_inches="tight")
print(f"\nSaved: {plot_path}")
plt.show()
