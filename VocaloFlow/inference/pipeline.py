"""End-to-end VocaloFlow inference pipeline.

Takes an OpenUTAU .ustx file (+ optional pre-rendered WAV and F0 curve)
and produces a high-quality singing voice WAV via:
  1. USTX parsing  ->  note/lyric/timing extraction
  2. Mel extraction from rendered prior WAV
  3. F0 extraction (RMVPE, provided .npy, or MIDI-pitch fallback)
  4. Frame-level phoneme ID construction
  5. Chunked ODE inference through VocaloFlow
  6. SoulX-Singer Vocos vocoding  ->  output WAV

Usage:
    cd VocaloFlow/
    python -m inference.pipeline --ustx prior.ustx --prior-wav prior.wav \
        --checkpoint checkpoints/checkpoint_200000.pt --output output.wav

python -m inference.pipeline --ustx "../demo/short_test.ustx" --prior-wav "../demo/short_test_Track1.wav" --checkpoint "checkpoints/checkpoint_20000.pt" --output "../demo/target_short_test.wav"
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup — allow imports from sibling project directories
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_VOCALOFLOW_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_VOCALOFLOW_DIR, ".."))
_DATASYNTHESIZER_DIR = os.path.join(_REPO_ROOT, "DataSynthesizer")
_SOULX_DIR = os.path.join(_REPO_ROOT, "SoulX-Singer")

# Add repo root to path for shared config
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Add VocaloFlow and SoulX-Singer to path (but NOT DataSynthesizer, whose
# "utils" package would shadow VocaloFlow's own "utils").
for _p in [_VOCALOFLOW_DIR, _SOULX_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# VocaloFlow imports
from configs.default import VocaloFlowConfig
from model.vocaloflow import VocaloFlow
from model.vocaloflow_wavenet import VocaloFlowPureWaveNet
from inference.inference import sample_ode
from utils.resample import resample_1d, resample_2d, resolve_phoneme_indirection

# ---------------------------------------------------------------------------
# DataSynthesizer imports — loaded via importlib to avoid "utils" namespace
# collision with VocaloFlow's own utils package.
# ---------------------------------------------------------------------------
import importlib.util as _ilu


def _import_from_path(module_name: str, file_path: str):
    spec = _ilu.spec_from_file_location(module_name, file_path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_vocoders_mod = _import_from_path(
    "ds_vocoders", os.path.join(_DATASYNTHESIZER_DIR, "utils", "vocoders.py")
)
mel_to_soulx_mel = _vocoders_mod.mel_to_soulx_mel
invert_mel_to_audio_soulxsinger = _vocoders_mod.invert_mel_to_audio_soulxsinger

_phoneme_mask_mod = _import_from_path(
    "ds_phoneme_mask", os.path.join(_DATASYNTHESIZER_DIR, "utils", "phoneme_mask.py")
)
_build_mel2note = _phoneme_mask_mod._build_mel2note
_load_phone2idx = _phoneme_mask_mod._load_phone2idx

_voiced_mod = _import_from_path(
    "ds_voiced_unvoiced", os.path.join(_DATASYNTHESIZER_DIR, "utils", "voiced_unvoiced.py")
)
get_voiced_mask = _voiced_mod.get_voiced_mask

# SoulX-Singer g2p (handles en_ prefixing and multi-language)
from preprocess.tools.g2p import g2p_transform, is_english_word

# ---------------------------------------------------------------------------
# Default paths (relative to repo root)
# ---------------------------------------------------------------------------
_DEFAULT_PHONESET = os.path.join(
    _SOULX_DIR, "soulxsinger", "utils", "phoneme", "phone_set.json"
)
_DEFAULT_RMVPE = os.path.join(
    _SOULX_DIR, "pretrained_models", "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt"
)

SR = 24000
HOP = 480

_PLBERT_DIR = os.path.join(_REPO_ROOT, "PL-BERT")
_DEFAULT_SPEAKER_EMB = os.path.join(
    _REPO_ROOT, "SpeakerEmbedding", "embeddings", "Rachie", "speaker_embedding.pt"
)


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Parse USTX
# ═══════════════════════════════════════════════════════════════════════════

def parse_ustx(ustx_path: str) -> dict:
    """Parse a .ustx YAML file and extract note information.

    Returns:
        dict with keys:
          - bpm (float)
          - resolution (int): ticks per beat
          - ms_per_tick (float)
          - notes: list of dicts with position_ticks, duration_ticks, tone,
                   lyric, pitch_points, snap_first, vibrato
          - pitd_curve: dict with xs/ys arrays (part-level pitch deviation), or None
    """
    import yaml

    with open(ustx_path, "r", encoding="utf-8") as f:
        project = yaml.safe_load(f)

    bpm = float(project.get("bpm", project.get("tempo", 120)))
    resolution = int(project.get("resolution", 480))
    ms_per_tick = (60_000.0 / bpm) / resolution

    # Find voice parts — may be under "voice_parts" or "parts"
    parts = project.get("voice_parts") or project.get("parts") or []
    if not parts:
        raise ValueError(f"No voice parts found in {ustx_path}")

    part = parts[0]
    raw_notes = part.get("notes", [])
    if not raw_notes:
        raise ValueError(f"No notes found in first voice part of {ustx_path}")

    notes = []
    for n in raw_notes:
        notes.append({
            "position_ticks": int(n["position"]),
            "duration_ticks": int(n["duration"]),
            "tone": int(n["tone"]),
            "lyric": str(n.get("lyric", "")),
            "pitch_points": n.get("pitch", {}).get("data", []),
            "snap_first": n.get("pitch", {}).get("snap_first", True),
            "vibrato": n.get("vibrato", {}),
        })

    # Part-level pitch deviation curve (usually empty)
    pitd_curve = None
    for c in part.get("curves", []):
        if c.get("abbr") == "pitd" and c.get("xs") and c.get("ys"):
            pitd_curve = {"xs": c["xs"], "ys": c["ys"]}
            break

    print(f"[pipeline] Parsed {len(notes)} notes from {os.path.basename(ustx_path)} "
          f"(bpm={bpm}, resolution={resolution})")
    return {
        "bpm": bpm,
        "resolution": resolution,
        "ms_per_tick": ms_per_tick,
        "notes": notes,
        "pitd_curve": pitd_curve,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Render or load prior WAV
# ═══════════════════════════════════════════════════════════════════════════

def render_ustx_to_wav(ustx_path: str, notes_data: dict, output_wav: str) -> str:
    """Render a USTX project to WAV using the C# Player API via pythonnet.

    Falls back with a clear error if pythonnet is not available.
    """
    try:
        from config import UTAU_GENERATE_DLL
        import pythonnet
        pythonnet.load("coreclr")
        import clr

        # Add DLL directory to PATH so native deps (e.g. onnxruntime.dll) are found
        bin_dir = os.path.dirname(UTAU_GENERATE_DLL)
        native_dir = os.path.join(bin_dir, "runtimes", "win-x64", "native")
        os.environ["PATH"] = bin_dir + os.pathsep + native_dir + os.pathsep + os.environ.get("PATH", "")

        clr.AddReference(UTAU_GENERATE_DLL)
        from UtauGenerate import Player
    except Exception as e:
        raise RuntimeError(
            f"Cannot render USTX without pythonnet and compiled UtauGenerate.dll.\n"
            f"Error: {e}\n"
            f"Please provide --prior-wav with a pre-rendered WAV of the USTX file."
        )

    player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
    for note in notes_data["notes"]:
        player.addNote(
            note["position_ticks"],
            note["duration_ticks"],
            note["tone"],
            note["lyric"],
        )

    player.validate()
    player.exportWav(output_wav)
    print(f"[pipeline] Rendered USTX to {output_wav}")
    return output_wav


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Extract prior mel
# ═══════════════════════════════════════════════════════════════════════════

def extract_prior_mel(wav_path: str) -> np.ndarray:
    """Extract SoulX-Singer-format normalized log-mel from a WAV file.

    Returns:
        np.ndarray of shape (T, 128).
    """
    import librosa
    y, _ = librosa.load(wav_path, sr=SR)
    mel = mel_to_soulx_mel(y, sr=SR)  # (128, T)
    mel = mel.T  # (T, 128)
    print(f"[pipeline] Extracted prior mel: {mel.shape} from {os.path.basename(wav_path)}")
    return mel.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Extract or load F0
# ═══════════════════════════════════════════════════════════════════════════

def extract_f0_rmvpe(wav_path: str, rmvpe_model_path: str, device: str) -> np.ndarray:
    """Extract F0 from a WAV file using the RMVPE model.

    Returns:
        np.ndarray of shape (T,), F0 in Hz (0 = unvoiced).
    """
    from preprocess.tools.f0_extraction import F0Extractor
    extractor = F0Extractor(
        model_path=rmvpe_model_path,
        device=device,
        target_sr=SR,
        hop_size=HOP,
    )
    f0 = extractor.process(wav_path)
    print(f"[pipeline] Extracted F0 via RMVPE: {len(f0)} frames, "
          f"voiced ratio={np.mean(f0 > 0):.2%}")
    return f0.astype(np.float32)


def synthesize_f0_from_midi(notes_data: dict, total_frames: int) -> np.ndarray:
    """Synthesize a coarse F0 curve from USTX MIDI pitches.

    Fallback when no F0 file or RMVPE model is available.
    """
    f0 = np.zeros(total_frames, dtype=np.float32)
    ms_per_tick = notes_data["ms_per_tick"]
    frame_dur_s = HOP / SR

    for note in notes_data["notes"]:
        start_s = note["position_ticks"] * ms_per_tick / 1000.0
        dur_s = note["duration_ticks"] * ms_per_tick / 1000.0
        start_frame = int(start_s / frame_dur_s)
        end_frame = int((start_s + dur_s) / frame_dur_s)
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        freq_hz = 440.0 * (2.0 ** ((note["tone"] - 69) / 12.0))
        f0[start_frame:end_frame] = freq_hz

    voiced = np.sum(f0 > 0)
    print(f"[pipeline] Synthesized F0 from MIDI pitches: {total_frames} frames, "
          f"voiced ratio={voiced / max(total_frames, 1):.2%}")
    return f0


def extract_or_load_f0(
    f0_path: str | None,
    wav_path: str,
    rmvpe_model_path: str,
    device: str,
    notes_data: dict,
    total_frames: int,
) -> np.ndarray:
    """Load F0 from file, extract via RMVPE, or synthesize from MIDI pitches."""
    if f0_path is not None:
        f0 = np.load(f0_path).astype(np.float32)
        print(f"[pipeline] Loaded F0 from {f0_path}: {len(f0)} frames")
        return f0

    if os.path.exists(rmvpe_model_path):
        try:
            return extract_f0_rmvpe(wav_path, rmvpe_model_path, device)
        except Exception as e:
            print(f"[pipeline] RMVPE extraction failed ({e}), falling back to MIDI pitch F0")

    return synthesize_f0_from_midi(notes_data, total_frames)


# ═══════════════════════════════════════════════════════════════════════════
# Step 4b: USTX pitch-curve F0 synthesis
# ═══════════════════════════════════════════════════════════════════════════

def _interpolate_pitch_points(pitch_points: list[dict], t_ms: float) -> float:
    """Linearly interpolate cents deviation from sorted pitch control points."""
    if not pitch_points:
        return 0.0
    if t_ms <= pitch_points[0]["x"]:
        return float(pitch_points[0]["y"])
    if t_ms >= pitch_points[-1]["x"]:
        return float(pitch_points[-1]["y"])
    for i in range(len(pitch_points) - 1):
        x0, y0 = pitch_points[i]["x"], pitch_points[i]["y"]
        x1, y1 = pitch_points[i + 1]["x"], pitch_points[i + 1]["y"]
        if x0 <= t_ms <= x1:
            alpha = (t_ms - x0) / (x1 - x0) if x1 != x0 else 0.0
            return y0 + alpha * (y1 - y0)
    return float(pitch_points[-1]["y"])


def synthesize_f0_from_ustx_pitch(
    notes_data: dict, total_frames: int, voicing: np.ndarray | None = None,
) -> np.ndarray:
    """Synthesize F0 from USTX per-note pitch control points.

    Each note has a base MIDI tone and pitch control points that define
    cents deviation over time.  The result is a frame-level F0 in Hz with
    unvoiced frames zeroed out (if *voicing* is provided).
    """
    f0 = np.zeros(total_frames, dtype=np.float32)
    ms_per_tick = notes_data["ms_per_tick"]
    frame_dur_s = HOP / SR
    frame_dur_ms = frame_dur_s * 1000.0

    for note in notes_data["notes"]:
        start_s = note["position_ticks"] * ms_per_tick / 1000.0
        dur_s = note["duration_ticks"] * ms_per_tick / 1000.0
        start_frame = int(start_s / frame_dur_s)
        end_frame = int((start_s + dur_s) / frame_dur_s)
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        base_hz = 440.0 * (2.0 ** ((note["tone"] - 69) / 12.0))
        pitch_points = note.get("pitch_points", [])

        for fi in range(start_frame, end_frame):
            t_ms = (fi - start_frame) * frame_dur_ms
            cents = _interpolate_pitch_points(pitch_points, t_ms)
            f0[fi] = base_hz * (2.0 ** (cents / 1200.0))

    if voicing is not None:
        f0[~voicing.astype(bool)] = 0.0

    voiced = np.sum(f0 > 0)
    print(f"[pipeline] Synthesized F0 from USTX pitch curves: {total_frames} frames, "
          f"voiced ratio={voiced / max(total_frames, 1):.2%}")
    return f0


# ═══════════════════════════════════════════════════════════════════════════
# Step 4c: Note-coverage voicing mask
# ═══════════════════════════════════════════════════════════════════════════

def get_note_coverage_voicing(
    notes_data: dict,
    total_frames: int,
) -> np.ndarray:
    """Derive a voiced/unvoiced mask from MIDI note coverage.

    A frame is voiced iff it falls within any note's time range.
    Frames in silence gaps between notes are unvoiced.
    """
    note_coverage = np.zeros(total_frames, dtype=bool)
    ms_per_tick = notes_data["ms_per_tick"]
    frame_dur_s = HOP / SR
    for note in notes_data["notes"]:
        start_s = note["position_ticks"] * ms_per_tick / 1000.0
        dur_s = note["duration_ticks"] * ms_per_tick / 1000.0
        s = max(0, min(int(start_s / frame_dur_s), total_frames))
        e = max(0, min(int((start_s + dur_s) / frame_dur_s), total_frames))
        note_coverage[s:e] = True

    voicing = note_coverage.astype(np.float32)
    voiced_count = int(np.sum(voicing > 0))
    print(f"[pipeline] Note-coverage voicing: voiced={voiced_count}/{total_frames} "
          f"({voiced_count / max(total_frames, 1) * 100:.1f}%)")
    return voicing


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Build per-frame phoneme IDs
# ═══════════════════════════════════════════════════════════════════════════

_BRACKET_RE = re.compile(r"^(.+?)\[.+\]$")


def _lyric_to_word(lyric: str) -> str:
    """Strip USTX bracket hints and handle continuation/rest notes.

    Returns the bare word for g2p, or ``"<SP>"`` for non-lyric notes.
    """
    lyric = lyric.strip()
    if lyric in ("+", "-", "R", "r", ""):
        return "<SP>"
    m = _BRACKET_RE.match(lyric)
    return m.group(1).strip() if m else lyric


def build_phoneme_ids(
    notes: list[dict],
    ms_per_tick: float,
    total_frames: int,
    phoneset_path: str,
) -> np.ndarray:
    """Build resolved per-frame phoneme token IDs from USTX notes.

    Uses SoulX-Singer's ``g2p_transform`` to convert lyrics to phoneme
    strings in the ``en_PHONE-PHONE`` format expected by ``phone_set.json``.

    Returns:
        np.ndarray of shape (total_frames,), int32 resolved phoneme IDs.
    """
    # Extract bare words from lyrics (strip hints, handle continuations)
    words = [_lyric_to_word(n["lyric"]) for n in notes]

    # g2p_transform converts English words -> "en_PH1-en_PH2-..." format,
    # passes through "<SP>" unchanged, and handles non-English gracefully.
    phoneme_strings = g2p_transform(words, lang="English")

    note_durations = [n["duration_ticks"] * ms_per_tick / 1000.0 for n in notes]

    # Build expanded phoneme IDs and frame-level mask
    phone2idx = _load_phone2idx(phoneset_path)
    phoneme_ids_arr, phoneme_mask_arr = _build_mel2note(
        note_durations, phoneme_strings, phone2idx, sr=SR, hop=HOP
    )

    if len(phoneme_mask_arr) == 0:
        print("[pipeline] WARNING: phoneme mask is empty, returning all <PAD>")
        return np.zeros(total_frames, dtype=np.int32)

    # Resolve indirection and resample to target length
    resolved = resolve_phoneme_indirection(phoneme_ids_arr, phoneme_mask_arr)
    resolved = resample_1d(resolved, total_frames, mode="nearest").numpy().astype(np.int32)

    print(f"[pipeline] Built phoneme IDs: {len(resolved)} frames, "
          f"vocab used: {len(np.unique(resolved))} unique tokens")
    return resolved


# ═══════════════════════════════════════════════════════════════════════════
# Step 5b: PL-BERT feature extraction (live, for inference)
# ═══════════════════════════════════════════════════════════════════════════

def build_plbert_frame_features(
    notes: list[dict],
    ms_per_tick: float,
    total_frames: int,
    phoneset_path: str,
    plbert_dir: str = _PLBERT_DIR,
    device: str = "cpu",
) -> np.ndarray:
    """Build per-frame PL-BERT features from USTX notes.

    Runs the same pipeline as precompute_features.py but on-the-fly:
    1. Build phoneme IDs from notes (same as build_phoneme_ids)
    2. Convert English phonemes to IPA
    3. Run PL-BERT for contextual embeddings
    4. Expand to frame-level

    Returns:
        (total_frames, 768) float32 array.
    """
    plbert_mod = _import_from_path("plbert", os.path.join(plbert_dir, "plbert.py"))
    text_utils = _import_from_path("text_utils", os.path.join(plbert_dir, "text_utils.py"))
    arpabet_ipa = _import_from_path("arpabet_ipa", os.path.join(plbert_dir, "arpabet_ipa.py"))

    import json
    with open(phoneset_path) as f:
        phone_set = json.load(f)

    extractor = plbert_mod.PLBertFeatureExtractor(device=device)

    # Build the expanded phoneme sequence (same as build_phoneme_ids)
    words = [_lyric_to_word(n["lyric"]) for n in notes]
    phoneme_strings = g2p_transform(words, lang="English")
    note_durations = [n["duration_ticks"] * ms_per_tick / 1000.0 for n in notes]
    phone2idx = _load_phone2idx(phoneset_path)

    phoneme_ids_arr, phoneme_mask_arr = _build_mel2note(
        note_durations, phoneme_strings, phone2idx, sr=SR, hop=HOP
    )

    if len(phoneme_mask_arr) == 0:
        return np.zeros((total_frames, 768), dtype=np.float32)

    # Extract PL-BERT features at phoneme level (P, 768)
    P = len(phoneme_ids_arr)
    features = np.zeros((P, 768), dtype=np.float32)

    en_positions = []
    for i, pid in enumerate(phoneme_ids_arr):
        entry = phone_set[pid]
        arpa = arpabet_ipa.phone_set_entry_to_arpabet(entry)
        if arpa is not None:
            en_positions.append((i, arpa))

    if en_positions:
        ipa_chars = []
        char_to_pidx = []
        for pidx, arpa in en_positions:
            ipa = arpabet_ipa.arpabet_to_ipa(arpa)
            if ipa is None:
                continue
            for c in ipa:
                ipa_chars.append(c)
                char_to_pidx.append(pidx)

        if ipa_chars:
            token_ids = text_utils.tokenize_ipa("".join(ipa_chars))
            plbert_out = extractor.extract(token_ids).cpu().numpy()

            from collections import Counter
            for pos, pidx in enumerate(char_to_pidx):
                features[pidx] += plbert_out[pos]
            counts = Counter(char_to_pidx)
            for pidx, count in counts.items():
                features[pidx] /= count

    # Expand to frame-level using phoneme_mask indirection
    resolved_feats = features[np.clip(phoneme_mask_arr, 0, P - 1)]
    resolved_feats = resample_2d(
        torch.from_numpy(resolved_feats), total_frames, mode="nearest"
    ).numpy()

    print(f"[pipeline] Built PL-BERT features: {resolved_feats.shape}, "
          f"non-zero frames: {np.any(resolved_feats != 0, axis=1).sum()}/{total_frames}")
    return resolved_feats.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Load model
# ═══════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load VocaloFlow from a training checkpoint (uses EMA weights)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", VocaloFlowConfig())
    arch = getattr(config, "architecture", "hybrid")
    if arch == "wavenet_pure":
        model = VocaloFlowPureWaveNet(config).to(device)
    else:
        model = VocaloFlow(config).to(device)

    state_key = "ema_model_state_dict" if "ema_model_state_dict" in ckpt else "model_state_dict"
    missing, unexpected = model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()

    step = ckpt.get("step", "?")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[pipeline] Loaded model from step {step} ({param_count:,} params, "
          f"using {state_key})")
    if missing:
        print(f"[pipeline] Randomly initialized keys (not in checkpoint): "
              f"{len(missing)} — {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[pipeline] Unexpected keys in checkpoint (ignored): "
              f"{len(unexpected)} — {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Step 7: Chunked ODE inference
# ═══════════════════════════════════════════════════════════════════════════

def _build_chunk_window(
    ci: int,
    start: int,
    end: int,
    length: int,
    total_frames: int,
    overlap: int,
    chunk_size: int,
    starts: list[int],
    tail_chunk_idx: int | None,
) -> np.ndarray:
    """Build the blending window for a single chunk.

    Regular chunks get linear fade-in at the left edge (if not the first
    chunk) and fade-out at the right edge (if not reaching the end).

    The tail chunk (appended to cover the sequence end) uses a window
    that is zero until the previous chunk's fade-out region, then
    crossfades over exactly *overlap* frames, then holds full weight to
    the end.  This keeps the crossfade clean and avoids the large
    averaging region that would result from a naive overlap.
    """
    if ci == tail_chunk_idx:
        prev_end = min(starts[ci - 1] + chunk_size, total_frames)
        xfade_start = prev_end - overlap - start
        xfade_end = prev_end - start

        window = np.zeros(length, dtype=np.float32)
        window[xfade_start:xfade_end] = np.linspace(0, 1, xfade_end - xfade_start)
        window[xfade_end:] = 1.0
        return window

    window = np.ones(length, dtype=np.float32)
    if start > 0 and overlap > 0:
        fade_len = min(overlap, length)
        window[:fade_len] = np.linspace(0, 1, fade_len)
    if end < total_frames and overlap > 0:
        fade_len = min(overlap, length)
        window[length - fade_len:] = np.linspace(1, 0, fade_len)
    return window


def infer_chunked(
    model: torch.nn.Module,
    prior_mel: np.ndarray,
    f0: np.ndarray,
    voicing: np.ndarray,
    phoneme_ids: np.ndarray,
    chunk_size: int = 256,
    overlap: int = 16,
    num_steps: int = 16,
    method: str = "midpoint",
    device: torch.device = torch.device("cpu"),
    cfg_scale: float = 1.0,
    plbert_features: np.ndarray | None = None,
    speaker_embedding: torch.Tensor | None = None,
    time_schedule: str = "sway",
    sway_coeff: float = -1.0,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> np.ndarray:
    """Run chunked ODE inference with overlap-add blending.

    For sequences shorter than *chunk_size*, pads to *chunk_size* and
    runs a single pass. For longer sequences, uses overlapping windows
    with linear crossfade.

    Returns:
        (T, 128) predicted high-quality mel-spectrogram.
    """
    T = prior_mel.shape[0]
    stride = max(1, chunk_size - overlap)

    # Build chunk start positions covering the full sequence.
    # If the last regular chunk doesn't reach T, append a tail chunk
    # starting at T-chunk_size.  Its blending window is handled specially
    # by _build_chunk_window so the crossfade aligns with the previous
    # chunk's fade-out (no large averaging region).
    starts = list(range(0, max(1, T - overlap), stride))
    tail_chunk_idx = None
    if starts[-1] + chunk_size < T:
        starts.append(max(0, T - chunk_size))
        tail_chunk_idx = len(starts) - 1

    output_mel = np.zeros((T, 128), dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)
    n_chunks = len(starts)

    print(f"[pipeline] Running inference: {n_chunks} chunk(s) "
          f"(T={T}, chunk={chunk_size}, overlap={overlap})")

    for ci, start in enumerate(starts):
        end = min(start + chunk_size, T)
        length = end - start

        # Slice inputs
        pm = prior_mel[start:end]
        f = f0[start:end]
        v = voicing[start:end]
        ph = phoneme_ids[start:end]
        pl = plbert_features[start:end] if plbert_features is not None else None

        # Pad to chunk_size if needed
        if length < chunk_size:
            pad_len = chunk_size - length
            pm = np.pad(pm, ((0, pad_len), (0, 0)))
            f = np.pad(f, (0, pad_len))
            v = np.pad(v, (0, pad_len))
            ph = np.pad(ph, (0, pad_len))
            if pl is not None:
                pl = np.pad(pl, ((0, pad_len), (0, 0)))

        # Padding mask and tensors
        mask = torch.zeros(1, chunk_size, dtype=torch.bool, device=device)
        mask[0, :length] = True

        pm_t = torch.from_numpy(pm.astype(np.float32)).unsqueeze(0).to(device)
        f_t = torch.from_numpy(f.astype(np.float32)).unsqueeze(0).to(device)
        v_t = torch.from_numpy(v.astype(np.float32)).unsqueeze(0).to(device)
        ph_t = torch.from_numpy(ph.astype(np.int64)).unsqueeze(0).to(device)
        pl_t = (torch.from_numpy(pl.astype(np.float32)).unsqueeze(0).to(device)
                if pl is not None else None)

        pred = sample_ode(model, pm_t, f_t, v_t, ph_t, num_steps, method, mask,
                          cfg_scale=cfg_scale, plbert_features=pl_t,
                          speaker_embedding=speaker_embedding,
                          time_schedule=time_schedule, sway_coeff=sway_coeff,
                          atol=atol, rtol=rtol)
        pred = pred[0, :length].cpu().numpy()

        window = _build_chunk_window(
            ci, start, end, length, T, overlap,
            chunk_size, starts, tail_chunk_idx,
        )

        output_mel[start:end] += pred * window[:, None]
        weight[start:end] += window

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci + 1}/{n_chunks}")

    output_mel /= np.maximum(weight[:, None], 1e-8)
    return output_mel


# ═══════════════════════════════════════════════════════════════════════════
# Step 8: Vocoder
# ═══════════════════════════════════════════════════════════════════════════

def mel_to_wav(mel: np.ndarray) -> np.ndarray:
    """Convert VocaloFlow output mel to audio via SoulX-Singer Vocos vocoder.

    Args:
        mel: (T, 128) normalized log-mel in SoulX-Singer space.

    Returns:
        1D numpy array of audio at 24kHz.
    """
    mel_transposed = mel.T.astype(np.float32)  # (128, T)
    audio = invert_mel_to_audio_soulxsinger(mel_transposed, config={})
    print(f"[pipeline] Vocoded mel to audio: {len(audio)} samples "
          f"({len(audio) / SR:.2f}s @ {SR}Hz)")
    return audio


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VocaloFlow end-to-end inference: USTX -> enhanced WAV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ustx", required=True, help="Path to .ustx file")
    p.add_argument("--checkpoint", required=True, help="Path to VocaloFlow .pt checkpoint")
    p.add_argument("--prior-wav", default=None,
                   help="Pre-rendered WAV of the USTX. If omitted, attempts C# Player API render.")
    p.add_argument("--f0", default=None,
                   help="Pre-extracted F0 .npy file (T,) in Hz. Implies --f0-mode file.")
    p.add_argument("--f0-mode", default="ustx",
                   choices=["ustx", "rmvpe", "midi", "file"],
                   help="F0/voicing extraction strategy: "
                        "ustx = pitch curves from USTX + phoneme voicing (default); "
                        "rmvpe = extract from prior WAV; "
                        "midi = flat MIDI pitches; "
                        "file = load from --f0 .npy")
    p.add_argument("--output", default="output.wav", help="Output WAV path")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Compute device")
    p.add_argument("--num-ode-steps", type=int, default=16, help="ODE integration steps")
    p.add_argument("--ode-method", default="midpoint",
                   choices=["euler", "midpoint", "rk4", "dopri5"],
                   help="ODE integration method")
    p.add_argument("--time-schedule", default="sway", choices=["uniform", "sway"],
                   help="Timestep distribution for ODE integration")
    p.add_argument("--sway-coeff", type=float, default=-1.0,
                   help="Sway coefficient for 'sway' time schedule")
    p.add_argument("--atol", type=float, default=1e-5,
                   help="Absolute tolerance for adaptive ODE methods (dopri5)")
    p.add_argument("--rtol", type=float, default=1e-5,
                   help="Relative tolerance for adaptive ODE methods (dopri5)")
    p.add_argument("--chunk-size", type=int, default=256,
                   help="Inference chunk size in frames (default 256 = 5.12s)")
    p.add_argument("--overlap", type=int, default=16,
                   help="Overlap frames for crossfade between chunks (16 = 0.32s)")
    p.add_argument("--rmvpe-model", default=_DEFAULT_RMVPE,
                   help="Path to RMVPE checkpoint for F0 extraction")
    p.add_argument("--phoneset", default=_DEFAULT_PHONESET,
                   help="Path to phone_set.json")
    p.add_argument("--cfg-scale", type=float, default=2.0,
                   help="Classifier-free guidance scale (1.0 = no guidance)")
    p.add_argument("--mask-phonemes", action="store_true",
                   help="Zero out all phoneme IDs (diagnostic: removes linguistic conditioning)")
    p.add_argument("--save-mels", action="store_true",
                   help="Save prior and output mel spectrograms as .npy alongside output WAV")
    p.add_argument("--plbert-dir", default=os.path.join(_REPO_ROOT, "PL-BERT"),
                   help="Path to PL-BERT directory (config, checkpoint, etc.)")
    p.add_argument("--speaker-embedding", default=_DEFAULT_SPEAKER_EMB,
                   help="Path to speaker embedding .pt file (default: Rachie)")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Conditioning extraction strategies
# ═══════════════════════════════════════════════════════════════════════════

def _extract_conditioning_ustx(
    notes_data: dict, T: int, phoneset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """USTX mode: voicing from note coverage -> F0 from pitch curves -> phonemes."""
    voicing = get_note_coverage_voicing(notes_data, T)
    f0 = synthesize_f0_from_ustx_pitch(notes_data, T, voicing=voicing)
    f0 = resample_1d(f0, T, mode="linear").numpy()
    phoneme_ids = build_phoneme_ids(
        notes_data["notes"], notes_data["ms_per_tick"], T, phoneset_path,
    )
    return f0, voicing, phoneme_ids


def _extract_conditioning_f0_based(
    f0_raw: np.ndarray,
    notes_data: dict,
    T: int,
    phoneset_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared path for modes where F0 is extracted first, voicing = f0 > 0."""
    f0 = resample_1d(f0_raw, T, mode="linear").numpy()
    voicing = get_voiced_mask(f0).astype(np.float32)
    phoneme_ids = build_phoneme_ids(
        notes_data["notes"], notes_data["ms_per_tick"], T, phoneset_path,
    )
    return f0, voicing, phoneme_ids


_F0_EXTRACTORS = {
    "file":  lambda args, **kw: np.load(args.f0).astype(np.float32),
    "rmvpe": lambda args, **kw: extract_f0_rmvpe(
        kw["wav_path"], args.rmvpe_model, kw["device"],
    ),
    "midi":  lambda args, **kw: synthesize_f0_from_midi(
        kw["notes_data"], kw["total_frames"],
    ),
}


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[pipeline] Device: {device}")

    # Step 1: Parse USTX
    notes_data = parse_ustx(args.ustx)

    # Step 2: Get prior WAV
    if args.prior_wav is not None:
        prior_wav = args.prior_wav
        if not os.path.exists(prior_wav):
            raise FileNotFoundError(f"Prior WAV not found: {prior_wav}")
    else:
        prior_wav = os.path.splitext(args.output)[0] + "_prior.wav"
        render_ustx_to_wav(args.ustx, notes_data, prior_wav)

    # Step 3: Extract prior mel
    prior_mel = extract_prior_mel(prior_wav)
    T = prior_mel.shape[0]

    # ── Diagnostic: prior mel stats ──
    print(f"[diag] prior_mel  shape={prior_mel.shape}  "
          f"min={prior_mel.min():.4f}  max={prior_mel.max():.4f}  "
          f"mean={prior_mel.mean():.4f}  std={prior_mel.std():.4f}")
    _pm_energy = np.mean(prior_mel ** 2, axis=1)
    _silent_frames = np.sum(_pm_energy < 1e-6)
    print(f"[diag] prior_mel  near-silent frames: {_silent_frames}/{T} "
          f"({_silent_frames / max(T, 1) * 100:.1f}%)")

    # Steps 4-6: Extract conditioning (f0, voicing, phoneme_ids)
    f0_mode = args.f0_mode
    if args.f0 is not None and f0_mode == "ustx":
        f0_mode = "file"
    print(f"[pipeline] F0 mode: {f0_mode}")

    if f0_mode == "ustx":
        f0, voicing, phoneme_ids = _extract_conditioning_ustx(
            notes_data, T, args.phoneset,
        )
    else:
        f0_raw = _F0_EXTRACTORS[f0_mode](
            args, wav_path=prior_wav, device=str(device),
            notes_data=notes_data, total_frames=T,
        )
        f0, voicing, phoneme_ids = _extract_conditioning_f0_based(
            f0_raw, notes_data, T, args.phoneset,
        )

    # ── Diagnostic: F0 and voicing stats ──
    print(f"[diag] f0         min={f0.min():.1f}  max={f0.max():.1f}  "
          f"mean(voiced)={f0[f0 > 0].mean() if np.any(f0 > 0) else 0:.1f} Hz")
    print(f"[diag] voicing    voiced={np.sum(voicing > 0)}/{T} "
          f"({np.mean(voicing > 0) * 100:.1f}%)")

    if args.mask_phonemes:
        phoneme_ids = np.zeros_like(phoneme_ids)
        print("[pipeline] --mask-phonemes set: zeroed out all phoneme IDs")

    # ── Diagnostic: phoneme ID stats ──
    _unique_ids = np.unique(phoneme_ids)
    _zero_pct = np.mean(phoneme_ids == 0) * 100
    print(f"[diag] phonemes   unique_tokens={len(_unique_ids)}  "
          f"zero/pad={_zero_pct:.1f}%  "
          f"ids_sample={_unique_ids[:10].tolist()}"
          f"{'...' if len(_unique_ids) > 10 else ''}")
    if _zero_pct > 90:
        print("[diag] WARNING: >90% of phoneme IDs are zero (padding). "
              "Model will receive almost no linguistic conditioning!")

    # Step 7: Load model
    model = load_model(args.checkpoint, device)

    # Step 7b: PL-BERT features (if model uses PL-BERT)
    plbert_features = None
    if model.config.use_plbert:
        print("[pipeline] Model uses PL-BERT — extracting features...")
        plbert_features = build_plbert_frame_features(
            notes_data["notes"], notes_data["ms_per_tick"], T,
            args.phoneset, plbert_dir=args.plbert_dir, device=str(device),
        )

    # Step 7c: Speaker embedding (if model uses it)
    speaker_embedding = None
    if model.config.use_speaker_embedding:
        spk_path = args.speaker_embedding
        if spk_path and os.path.exists(spk_path):
            speaker_embedding = torch.load(spk_path, weights_only=True).float().unsqueeze(0).to(device)
            print(f"[pipeline] Loaded speaker embedding from {spk_path}: {speaker_embedding.shape}")
        else:
            print(f"[pipeline] WARNING: use_speaker_embedding=True but no embedding found at {spk_path}")

    # Step 8: Chunked inference
    output_mel = infer_chunked(
        model, prior_mel, f0, voicing, phoneme_ids,
        chunk_size=args.chunk_size, overlap=args.overlap,
        num_steps=args.num_ode_steps, method=args.ode_method,
        device=device, cfg_scale=args.cfg_scale,
        plbert_features=plbert_features,
        speaker_embedding=speaker_embedding,
        time_schedule=args.time_schedule, sway_coeff=args.sway_coeff,
        atol=args.atol, rtol=args.rtol,
    )

    # ── Diagnostic: output mel vs prior mel ──
    print(f"[diag] output_mel shape={output_mel.shape}  "
          f"min={output_mel.min():.4f}  max={output_mel.max():.4f}  "
          f"mean={output_mel.mean():.4f}  std={output_mel.std():.4f}")
    _mel_diff = np.abs(output_mel - prior_mel)
    print(f"[diag] |output - prior|  mean={_mel_diff.mean():.4f}  "
          f"max={_mel_diff.max():.4f}  "
          f"(0 = model made no change)")
    _out_energy = np.mean(output_mel ** 2, axis=1)
    _out_silent = np.sum(_out_energy < 1e-6)
    print(f"[diag] output_mel near-silent frames: {_out_silent}/{T} "
          f"({_out_silent / max(T, 1) * 100:.1f}%)")

    # Ensure output directory exists (needed for both mel saves and final WAV)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Save mel spectrograms if requested
    if args.save_mels:
        stem = os.path.splitext(os.path.abspath(args.output))[0]
        np.save(f"{stem}_prior_mel.npy", prior_mel)
        np.save(f"{stem}_mel.npy", output_mel)
        print(f"[pipeline] Saved mel spectrograms: {stem}_prior_mel.npy, {stem}_mel.npy")

    # Step 9: Vocoder
    audio = mel_to_wav(output_mel)

    # ── Diagnostic: audio stats ──
    print(f"[diag] audio      min={audio.min():.6f}  max={audio.max():.6f}  "
          f"rms={np.sqrt(np.mean(audio ** 2)):.6f}  "
          f"peak_db={20 * np.log10(max(np.abs(audio).max(), 1e-10)):.1f} dBFS")

    # Step 10: Save output
    import soundfile as sf
    sf.write(args.output, audio, SR)
    print(f"[pipeline] Saved output to {args.output} "
          f"({len(audio) / SR:.2f}s @ {SR}Hz)")


if __name__ == "__main__":
    main()
