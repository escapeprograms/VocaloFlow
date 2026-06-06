"""Modular evaluation pipeline for VocaloFlow singing voice synthesis.

Usage:
    cd "Honors Thesis"
    python -m Evaluation.run_eval \\
        --checkpoint VocaloFlow/checkpoints/my-run/checkpoint_200000.pt

    # Quick smoke test (mel-domain only, 1 chunk)
    python -m Evaluation.run_eval --checkpoint ... \\
        --max-eval-chunks 1 --disable-wer --disable-mos

    # Both contexts with subset
    python -m Evaluation.run_eval --checkpoint ... \\
        --eval-context both --max-eval-chunks 50

    # From YAML config
    python -m Evaluation.run_eval --config Evaluation/configs/my_eval.yaml
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from eval_config import EvalConfig
from eval_utils.imports import (
    DATASYNTHESIZER_DIR,
    VOCALOFLOW_DIR,
    import_from_path,
    setup_vocaloflow_sys_path,
    timestamp,
)
from eval_utils.val_set import get_chunk_dir, get_val_manifest

# ---------------------------------------------------------------------------
# Cross-module imports (targeted — avoids pipeline.py's heavy g2p deps)
# ---------------------------------------------------------------------------
setup_vocaloflow_sys_path()

from configs.default import VocaloFlowConfig
from model.vocaloflow_hybrid import VocaloFlowHybrid
from model.vocaloflow_wavenet import VocaloFlowWaveNet
from inference.inference import sample_ode
from utils.resample import resample_1d, resample_2d, resolve_phoneme_indirection

SR = 24000
HOP = 480


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load VocaloFlow from a training checkpoint (uses EMA weights)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", VocaloFlowConfig())
    arch = getattr(config, "architecture", "hybrid")
    if arch == "wavenet_pure":
        model = VocaloFlowWaveNet(config).to(device)
    else:
        model = VocaloFlowHybrid(config).to(device)

    state_key = "ema_model_state_dict" if "ema_model_state_dict" in ckpt else "model_state_dict"
    model.load_state_dict(ckpt[state_key], strict=False)
    model.eval()

    step = ckpt.get("step", "?")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[{timestamp()}] Loaded model from step {step} ({param_count:,} params, {state_key})")
    return model


def _build_chunk_window(ci, start, end, length, total_frames, overlap, chunk_size, starts, tail_chunk_idx):
    """Build blending window for a single chunk (copied from pipeline.py)."""
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
    model, prior_mel, f0, voicing, phoneme_ids,
    chunk_size=256, overlap=16, num_steps=16, method="midpoint",
    device=torch.device("cpu"), cfg_scale=1.0,
    plbert_features=None, speaker_embedding=None,
    time_schedule="sway", sway_coeff=-1.0, atol=1e-5, rtol=1e-5,
):
    """Run chunked ODE inference with overlap-add blending."""
    T = prior_mel.shape[0]
    stride = max(1, chunk_size - overlap)

    starts = list(range(0, max(1, T - overlap), stride))
    tail_chunk_idx = None
    if starts[-1] + chunk_size < T:
        starts.append(max(0, T - chunk_size))
        tail_chunk_idx = len(starts) - 1

    output_mel = np.zeros((T, 128), dtype=np.float32)
    weight = np.zeros(T, dtype=np.float32)
    n_chunks = len(starts)

    print(f"[{timestamp()}] Inference: {n_chunks} chunk(s) (T={T}, chunk={chunk_size}, overlap={overlap})")

    for ci, start in enumerate(starts):
        end = min(start + chunk_size, T)
        length = end - start

        pm = prior_mel[start:end]
        f = f0[start:end]
        v = voicing[start:end]
        ph = phoneme_ids[start:end]
        pl = plbert_features[start:end] if plbert_features is not None else None

        if length < chunk_size:
            pad_len = chunk_size - length
            pm = np.pad(pm, ((0, pad_len), (0, 0)))
            f = np.pad(f, (0, pad_len))
            v = np.pad(v, (0, pad_len))
            ph = np.pad(ph, (0, pad_len))
            if pl is not None:
                pl = np.pad(pl, ((0, pad_len), (0, 0)))

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

        window = _build_chunk_window(ci, start, end, length, T, overlap, chunk_size, starts, tail_chunk_idx)
        output_mel[start:end] += pred * window[:, None]
        weight[start:end] += window

        if (ci + 1) % 10 == 0 or ci == n_chunks - 1:
            print(f"  chunk {ci + 1}/{n_chunks}")

    output_mel /= np.maximum(weight[:, None], 1e-8)
    return output_mel


# ═══════════════════════════════════════════════════════════════════════════
# Data loading (per-chunk, full length — no cropping to max_seq_len)
# ═══════════════════════════════════════════════════════════════════════════

def _load_chunk_arrays(row: dict, config: EvalConfig) -> dict:
    """Load all pre-computed arrays for a single chunk.

    Returns numpy arrays at their natural lengths (no cropping/padding).
    """
    prior_mel = np.load(row["prior_mel_path"]).astype(np.float32)
    target_mel = np.load(row["target_mel_path"]).astype(np.float32)
    f0 = np.load(row["f0_path"]).astype(np.float32)
    voicing = np.load(row["voicing_path"]).astype(np.float32)

    phoneme_mask = np.load(row["phoneme_mask_path"]).astype(np.int64)
    chunk_dir = os.path.dirname(row["phoneme_mask_path"])
    phoneme_ids = np.load(os.path.join(chunk_dir, "phoneme_ids.npy")).astype(np.int64)
    resolved = resolve_phoneme_indirection(phoneme_ids, phoneme_mask)

    T = prior_mel.shape[0]
    if len(f0) != T:
        f0 = np.pad(f0, (0, max(0, T - len(f0))))[:T]
    if len(voicing) != T:
        voicing = np.pad(voicing, (0, max(0, T - len(voicing))))[:T]
    if len(resolved) != T:
        resolved = np.pad(resolved, (0, max(0, T - len(resolved))))[:T]

    result = {
        "prior_mel": prior_mel,
        "target_mel": target_mel,
        "f0": f0,
        "voicing": voicing,
        "phoneme_ids": resolved.astype(np.int64),
        "chunk_dir": chunk_dir,
    }

    # PL-BERT (optional)
    plbert_path = os.path.join(chunk_dir, "plbert_features.npy")
    if os.path.exists(plbert_path):
        plbert_feats = np.load(plbert_path).astype(np.float32)
        mask_clipped = np.clip(phoneme_mask, 0, len(plbert_feats) - 1)
        plbert_frame = plbert_feats[mask_clipped]
        if len(plbert_frame) != T:
            if len(plbert_frame) < T:
                plbert_frame = np.pad(
                    plbert_frame,
                    ((0, T - len(plbert_frame)), (0, 0)),
                )
            else:
                plbert_frame = plbert_frame[:T]
        result["plbert_features"] = plbert_frame
    else:
        result["plbert_features"] = None

    return result


def _load_speaker_embedding(
    config: EvalConfig, device: torch.device
) -> torch.Tensor | None:
    """Load global speaker embedding if the checkpoint config requires it."""
    if not config.speaker_embedding_path or not os.path.exists(config.speaker_embedding_path):
        return None
    return torch.load(config.speaker_embedding_path, weights_only=True).float().unsqueeze(0).to(device)


# ═══════════════════════════════════════════════════════════════════════════
# Per-chunk evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _eval_quality(
    arrays: dict,
    model: torch.nn.Module,
    config: EvalConfig,
    device: torch.device,
    speaker_emb: torch.Tensor | None,
    rvc_audio_path: str | None,
    dali_id: str = "",
    chunk_name: str = "",
) -> dict:
    """Run Context 1 (quality) evaluation for a single chunk.

    Returns a dict of metric name → value.
    """
    from eval_utils.audio import extract_f0, vocode_mel, unload_f0_extractor, free_vram
    from metrics import mcd as mcd_mod
    from metrics import f0_metrics as f0_mod
    from metrics import wer as wer_mod
    from metrics import mos as mos_mod
    from metrics import contentvec as cv_mod

    prior_mel = arrays["prior_mel"]
    target_mel = arrays["target_mel"]
    target_f0 = arrays["f0"]
    chunk_dir = arrays["chunk_dir"]

    # ── VocaloFlow inference ─────────────────────────────────────────
    pred_mel = infer_chunked(
        model,
        prior_mel,
        arrays["f0"],
        arrays["voicing"],
        arrays["phoneme_ids"],
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        num_steps=config.num_ode_steps,
        method=config.ode_method,
        device=device,
        cfg_scale=config.cfg_scale,
        plbert_features=arrays["plbert_features"],
        speaker_embedding=speaker_emb,
    )

    results: dict = {}
    ref_lyrics = ""

    # ── MCD (mel-domain, no vocoding) ────────────────────────────────
    if config.enable_mcd:
        results["mcd"] = mcd_mod.compute_mcd(pred_mel, target_mel, config.mcd_n_mfcc)

    # ── Audio-domain metrics ─────────────────────────────────────────
    pred_audio = None
    target_audio = None
    pred_f0 = None

    if config.needs_audio():
        pred_audio = vocode_mel(pred_mel)

        target_wav_path = os.path.join(chunk_dir, "target.wav")
        if os.path.exists(target_wav_path):
            import librosa
            target_audio, _ = librosa.load(target_wav_path, sr=config.sr)
        else:
            target_audio = vocode_mel(target_mel)

    if config.needs_f0_extraction() and pred_audio is not None:
        pred_f0 = extract_f0(pred_audio, config.rmvpe_model_path, str(device), config.sr, config.hop)

    # ── F0 metrics (pred vs conditioning F0) ─────────────────────────
    if config.enable_f0_rmse and pred_f0 is not None:
        results["f0_rmse"] = f0_mod.f0_rmse(pred_f0, target_f0)

    if config.enable_f0_pearson and pred_f0 is not None:
        results["f0_pearson"] = f0_mod.f0_pearson(pred_f0, target_f0)

    if config.enable_ffe_gpe_vde and pred_f0 is not None:
        gvf = f0_mod.compute_gpe_vde_ffe(pred_f0, target_f0)
        results["gpe"] = gvf["gpe"]
        results["vde"] = gvf["vde"]
        results["ffe"] = gvf["ffe"]

    # Free RMVPE + ODE intermediates before loading Whisper (~3GB)
    unload_f0_extractor()
    free_vram()

    # ── WER (both systems vs DALI ground-truth lyrics) ─────────────
    if config.enable_wer and pred_audio is not None:
        from eval_utils.dali_ref import (
            extract_dali_lyrics,
            get_chunk_time_range,
            load_dali_entry,
        )

        ref_lyrics = ""
        dali_gz = os.path.join(config.dali_annot_dir, f"{dali_id}.gz")
        if dali_id and os.path.exists(dali_gz):
            try:
                dali_entry = load_dali_entry(config.dali_annot_dir, dali_id)
                chunk_start, chunk_end = get_chunk_time_range(dali_entry, chunk_name)
                ref_lyrics = extract_dali_lyrics(dali_entry, chunk_start, chunk_end)
            except (IndexError, ValueError):
                pass

        if ref_lyrics:
            pred_text = wer_mod.transcribe(pred_audio, config.sr, config.whisper_model, str(device))
            results["wer_vf"] = wer_mod.compute_wer(pred_text, ref_lyrics)

            if target_audio is not None:
                target_text = wer_mod.transcribe(target_audio, config.sr, config.whisper_model, str(device))
                results["wer_sx"] = wer_mod.compute_wer(target_text, ref_lyrics)

            prior_wav_path = os.path.join(chunk_dir, "prior.wav")
            if os.path.exists(prior_wav_path):
                import librosa
                prior_audio_wer, _ = librosa.load(prior_wav_path, sr=config.sr)
                prior_text = wer_mod.transcribe(prior_audio_wer, config.sr, config.whisper_model, str(device))
                results["wer_teto"] = wer_mod.compute_wer(prior_text, ref_lyrics)

    # Free Whisper before loading UTMOS
    wer_mod.unload()
    free_vram()

    # ── MOS ──────────────────────────────────────────────────────────
    if config.enable_mos and pred_audio is not None:
        results["mos_pred"] = mos_mod.compute_mos(pred_audio, config.sr, str(device))
        if target_audio is not None:
            results["mos_target"] = mos_mod.compute_mos(target_audio, config.sr, str(device))
        prior_wav_path = os.path.join(chunk_dir, "prior.wav")
        if os.path.exists(prior_wav_path):
            import librosa
            prior_audio, _ = librosa.load(prior_wav_path, sr=config.sr)
            results["mos_prior"] = mos_mod.compute_mos(prior_audio, config.sr, str(device))

    # ── ContentVec ───────────────────────────────────────────────────
    if config.enable_contentvec and pred_audio is not None and target_audio is not None:
        pred_cv = cv_mod.extract_features(pred_audio, config.sr, str(device))
        target_cv = cv_mod.extract_features(target_audio, config.sr, str(device))
        sim = cv_mod.compute_cosine_similarity(pred_cv, target_cv)
        results["contentvec_cosine_mean"] = sim["cosine_mean"]
        results["contentvec_cosine_std"] = sim["cosine_std"]

        prior_wav_path = os.path.join(chunk_dir, "prior.wav")
        if os.path.exists(prior_wav_path):
            import librosa
            prior_audio, _ = librosa.load(prior_wav_path, sr=config.sr)
            prior_cv = cv_mod.extract_features(prior_audio, config.sr, str(device))
            sim_prior = cv_mod.compute_cosine_similarity(prior_cv, target_cv)
            results["contentvec_prior_cosine_mean"] = sim_prior["cosine_mean"]

    # Free MOS/ContentVec models before RVC section
    mos_mod.unload()
    free_vram()

    # ── RVC baseline (if provided) ───────────────────────────────────
    if rvc_audio_path and os.path.exists(rvc_audio_path):
        import librosa
        rvc_audio, _ = librosa.load(rvc_audio_path, sr=config.sr)
        rvc_f0 = extract_f0(rvc_audio, config.rmvpe_model_path, str(device), config.sr, config.hop)

        if config.enable_f0_rmse:
            results["rvc_f0_rmse"] = f0_mod.f0_rmse(rvc_f0, target_f0)
        if config.enable_f0_pearson:
            results["rvc_f0_pearson"] = f0_mod.f0_pearson(rvc_f0, target_f0)
        if config.enable_ffe_gpe_vde:
            rvc_gvf = f0_mod.compute_gpe_vde_ffe(rvc_f0, target_f0)
            results["rvc_gpe"] = rvc_gvf["gpe"]
            results["rvc_vde"] = rvc_gvf["vde"]
            results["rvc_ffe"] = rvc_gvf["ffe"]
        if config.enable_mcd:
            from eval_utils.audio import audio_to_mel
            rvc_mel = audio_to_mel(rvc_audio, config.sr)
            results["rvc_mcd"] = mcd_mod.compute_mcd(rvc_mel, target_mel)
        if config.enable_wer and ref_lyrics:
            rvc_text = wer_mod.transcribe(rvc_audio, config.sr, config.whisper_model, str(device))
            results["rvc_wer"] = wer_mod.compute_wer(rvc_text, ref_lyrics)
        if config.enable_mos:
            results["rvc_mos"] = mos_mod.compute_mos(rvc_audio, config.sr, str(device))

    # Final cleanup — free everything before next context
    wer_mod.unload()
    mos_mod.unload()
    unload_f0_extractor()
    free_vram()

    return results


# ─── Context 2: Controllability (DALI-path USTX priors) ──────────────────

_pipeline_mod = None


def _get_pipeline_mod():
    """Lazy-load VocaloFlow inference/pipeline.py (heavy g2p deps)."""
    global _pipeline_mod
    if _pipeline_mod is None:
        _pipeline_mod = import_from_path(
            "vf_pipeline",
            os.path.join(VOCALOFLOW_DIR, "inference", "pipeline.py"),
        )
    return _pipeline_mod


def _eval_controllability(
    dali_id: str,
    chunk_name: str,
    arrays: dict,
    model: torch.nn.Module,
    config: EvalConfig,
    device: torch.device,
    speaker_emb: torch.Tensor | None,
    rvc_dali_audio_path: str | None = None,
) -> dict | None:
    """Run Context 2 (controllability) evaluation for a single chunk.

    Loads pre-generated DALI USTX + prior WAV, extracts voicing and phonemes
    from USTX notes, and uses DALI ground-truth F0 as the pitch conditioning.
    Runs VF inference and compares the output F0 against the same DALI reference.
    """
    from eval_utils.dali_ref import (
        extract_dali_f0_reference,
        extract_dali_note_timings,
        get_chunk_time_range,
        load_dali_entry,
    )
    from eval_utils.audio import extract_f0, vocode_mel
    from metrics import f0_metrics as f0_mod
    from metrics import duration as dur_mod

    # ── Locate pre-generated artifacts ───────────────────────────────
    chunk_dir = os.path.join(config.dali_ustx_dir, dali_id, chunk_name)
    ustx_path = os.path.join(chunk_dir, "dali_prior.ustx")
    wav_path = os.path.join(chunk_dir, "dali_prior.wav")

    if not os.path.exists(ustx_path) or not os.path.exists(wav_path):
        return None

    # ── Load DALI reference ──────────────────────────────────────────
    dali_gz = os.path.join(config.dali_annot_dir, f"{dali_id}.gz")
    if not os.path.exists(dali_gz):
        return None

    try:
        dali_entry = load_dali_entry(config.dali_annot_dir, dali_id)
        chunk_start, chunk_end = get_chunk_time_range(dali_entry, chunk_name)
    except (IndexError, ValueError):
        return None

    # ── Parse USTX and extract prior mel ────────────────────────────
    pipe = _get_pipeline_mod()
    notes_data = pipe.parse_ustx(ustx_path)
    prior_mel = pipe.extract_prior_mel(wav_path)
    T = prior_mel.shape[0]

    # ── Extract conditioning: voicing + phonemes from USTX notes,
    #    F0 from DALI ground-truth annotations ────────────────────────
    voicing = pipe.get_note_coverage_voicing(notes_data, T)
    phoneme_ids = pipe.build_phoneme_ids(
        notes_data["notes"], notes_data["ms_per_tick"], T,
        config.phoneset_path,
    )
    dali_f0 = extract_dali_f0_reference(
        dali_entry, chunk_start, chunk_end, T, config.sr, config.hop,
    )

    # ── PL-BERT features (conditional) ──────────────────────────────
    plbert_features = None
    uses_plbert = getattr(model, "config", None) and getattr(model.config, "use_plbert", False)
    if uses_plbert:
        try:
            plbert_features = pipe.build_plbert_frame_features(
                notes_data["notes"], notes_data["ms_per_tick"], T,
                config.phoneset_path, device=str(device),
            )
        except (ModuleNotFoundError, ImportError) as e:
            print(f"  PL-BERT live extraction failed ({e}), attempting precomputed fallback...")
            sx_plbert = arrays.get("plbert_features")
            if sx_plbert is not None:
                plbert_features = resample_2d(
                    torch.from_numpy(sx_plbert), T, mode="nearest",
                ).numpy().astype(np.float32)
                print(f"  PL-BERT fallback: resampled to {plbert_features.shape}")
            else:
                print("  WARNING: No precomputed PL-BERT features available, skipping")

    # ── VocaloFlow inference with DALI F0 conditioning ──────────────
    pred_mel = infer_chunked(
        model,
        prior_mel,
        dali_f0,
        voicing,
        phoneme_ids,
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        num_steps=config.num_ode_steps,
        method=config.ode_method,
        device=device,
        cfg_scale=config.cfg_scale,
        plbert_features=plbert_features,
        speaker_embedding=speaker_emb,
    )

    pred_audio = vocode_mel(pred_mel)
    pred_f0 = extract_f0(
        pred_audio, config.rmvpe_model_path, str(device), config.sr, config.hop,
    )

    results: dict = {}

    # ── VF (DALI-conditioned) F0 vs DALI score ───────────────────────
    results["vf_f0_rmse"] = f0_mod.f0_rmse(pred_f0, dali_f0)
    results["vf_f0_pearson"] = f0_mod.f0_pearson(pred_f0, dali_f0)
    gvf = f0_mod.compute_gpe_vde_ffe(pred_f0, dali_f0)
    results["vf_gpe"] = gvf["gpe"]
    results["vf_vde"] = gvf["vde"]
    results["vf_ffe"] = gvf["ffe"]

    # ── SoulX target F0 vs DALI score (for side-by-side comparison) ──
    # Build a separate DALI reference sized to SX's frame count, since the
    # DALI USTX prior has different duration than the SoulX-aligned prior.
    target_f0 = arrays["f0"]
    T_sx = len(target_f0)
    dali_f0_sx = extract_dali_f0_reference(
        dali_entry, chunk_start, chunk_end, T_sx, config.sr, config.hop,
    )
    results["sx_f0_rmse"] = f0_mod.f0_rmse(target_f0, dali_f0_sx)
    results["sx_f0_pearson"] = f0_mod.f0_pearson(target_f0, dali_f0_sx)
    gvf_sx = f0_mod.compute_gpe_vde_ffe(target_f0, dali_f0_sx)
    results["sx_gpe"] = gvf_sx["gpe"]
    results["sx_vde"] = gvf_sx["vde"]
    results["sx_ffe"] = gvf_sx["ffe"]

    # ── Onset reference (shared by VF, SX, and RVC) ─────────────────
    dali_onsets = np.array([], dtype=np.float32)
    if config.enable_duration_mae:
        dali_notes = extract_dali_note_timings(dali_entry, chunk_start, chunk_end)
        dali_onsets = dur_mod.extract_dali_onset_times(dali_notes)

    # ── Onset MAE (VF + SX) ──────────────────────────────────────────
    if config.enable_duration_mae and len(dali_onsets) > 0:
        vf_onsets = dur_mod.detect_note_onsets(pred_f0, config.sr, config.hop)
        results["vf_onset_mae"] = dur_mod.compute_onset_mae(vf_onsets, dali_onsets)
        sx_onsets = dur_mod.detect_note_onsets(target_f0, config.sr, config.hop)
        results["sx_onset_mae"] = dur_mod.compute_onset_mae(sx_onsets, dali_onsets)

    # ── RVC baseline: DALI prior → RVC → F0 vs DALI ────────────────
    if rvc_dali_audio_path and os.path.exists(rvc_dali_audio_path):
        import librosa
        rvc_audio, _ = librosa.load(rvc_dali_audio_path, sr=config.sr)
        rvc_f0 = extract_f0(
            rvc_audio, config.rmvpe_model_path, str(device), config.sr, config.hop,
        )

        results["rvc_f0_rmse"] = f0_mod.f0_rmse(rvc_f0, dali_f0)
        results["rvc_f0_pearson"] = f0_mod.f0_pearson(rvc_f0, dali_f0)
        rvc_gvf = f0_mod.compute_gpe_vde_ffe(rvc_f0, dali_f0)
        results["rvc_gpe"] = rvc_gvf["gpe"]
        results["rvc_vde"] = rvc_gvf["vde"]
        results["rvc_ffe"] = rvc_gvf["ffe"]

        if config.enable_duration_mae and len(dali_onsets) > 0:
            rvc_onsets = dur_mod.detect_note_onsets(rvc_f0, config.sr, config.hop)
            results["rvc_onset_mae"] = dur_mod.compute_onset_mae(rvc_onsets, dali_onsets)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Aggregation and output
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate(records: list[dict], label: str) -> pd.DataFrame:
    """Build per-sample DataFrame and print summary."""
    if not records:
        print(f"[{timestamp()}] No records for {label}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n{'=' * 60}")
    print(f"  {label} Summary  ({len(df)} samples)")
    print(f"{'=' * 60}")
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) == 0:
            continue
        print(f"  {col:30s}  mean={vals.mean():8.4f}  std={vals.std():8.4f}  "
              f"median={vals.median():8.4f}")
    print(f"{'=' * 60}\n")

    return df


def _save_results(
    df: pd.DataFrame, output_dir: str, name: str
) -> None:
    """Save per-sample CSV and summary CSV."""
    os.makedirs(output_dir, exist_ok=True)

    per_sample_path = os.path.join(output_dir, f"per_sample_{name}.csv")
    df.to_csv(per_sample_path, index=False)
    print(f"[{timestamp()}] Saved {per_sample_path}")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary = df[numeric_cols].agg(["mean", "std", "median", "min", "max"]).T
        summary_path = os.path.join(output_dir, f"summary_{name}.csv")
        summary.to_csv(summary_path)
        print(f"[{timestamp()}] Saved {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI & main
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="VocaloFlow Evaluation Pipeline")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--max-eval-chunks", type=int, default=None)
    parser.add_argument("--eval-context", type=str, default=None,
                        choices=["quality", "controllability", "both"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-ode-steps", type=int, default=None)
    parser.add_argument("--cfg-scale", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--whisper-model", type=str, default=None)
    parser.add_argument("--rvc-audio-dir", type=str, default=None)
    parser.add_argument("--rvc-dali-audio-dir", type=str, default=None,
                        help="Directory with pre-generated RVC audio from DALI priors (Context 2)")
    parser.add_argument("--dali-ustx-dir", type=str, default=None,
                        help="Directory with pre-generated DALI USTX priors")

    for metric in ["f0_rmse", "f0_pearson", "ffe_gpe_vde", "mcd", "wer", "mos",
                    "duration_mae", "contentvec"]:
        flag = metric.replace("_", "-")
        parser.add_argument(f"--enable-{flag}", action="store_true", default=None)
        parser.add_argument(f"--disable-{flag}", action="store_true", default=None)

    args = parser.parse_args()

    config = EvalConfig()
    if args.config:
        config = EvalConfig.from_yaml(args.config)

    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.manifest:
        config.manifest_path = args.manifest
    if args.max_eval_chunks is not None:
        config.max_eval_chunks = args.max_eval_chunks
    if args.eval_context:
        config.eval_context = args.eval_context
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_ode_steps is not None:
        config.num_ode_steps = args.num_ode_steps
    if args.cfg_scale is not None:
        config.cfg_scale = args.cfg_scale
    if args.device:
        config.device = args.device
    if args.whisper_model:
        config.whisper_model = args.whisper_model
    if args.rvc_audio_dir:
        config.rvc_audio_dir = args.rvc_audio_dir
    if args.rvc_dali_audio_dir:
        config.rvc_dali_audio_dir = args.rvc_dali_audio_dir
    if args.dali_ustx_dir:
        config.dali_ustx_dir = args.dali_ustx_dir

    for metric in ["f0_rmse", "f0_pearson", "ffe_gpe_vde", "mcd", "wer", "mos",
                    "duration_mae", "contentvec"]:
        enable_flag = getattr(args, f"enable_{metric}", None)
        disable_flag = getattr(args, f"disable_{metric}", None)
        if enable_flag:
            setattr(config, f"enable_{metric}", True)
        if disable_flag:
            setattr(config, f"enable_{metric}", False)

    return config


def main() -> None:
    config = _parse_args()

    if not config.checkpoint_path:
        print("Error: --checkpoint is required")
        sys.exit(1)

    # ── Device ───────────────────────────────────────────────────────
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"[{timestamp()}] Device: {device}")

    # ── Load model ───────────────────────────────────────────────────
    print(f"[{timestamp()}] Loading model from {config.checkpoint_path}")
    model = load_model(config.checkpoint_path, device)

    # Check if model config needs PL-BERT or speaker embedding
    ckpt = torch.load(config.checkpoint_path, map_location="cpu", weights_only=False)
    model_config = ckpt.get("config", None)
    uses_plbert = getattr(model_config, "use_plbert", False) if model_config else False
    uses_speaker = getattr(model_config, "use_speaker_embedding", False) if model_config else False
    del ckpt

    speaker_emb = None
    if uses_speaker:
        speaker_emb = _load_speaker_embedding(config, device)
        if speaker_emb is not None:
            print(f"[{timestamp()}] Loaded speaker embedding: {speaker_emb.shape}")

    # ── Load validation set ──────────────────────────────────────────
    print(f"[{timestamp()}] Loading validation set")
    val_df = get_val_manifest(
        config.data_dir,
        config.manifest_path,
        config.max_dtw_cost,
        config.val_fraction,
        config.seed,
        config.max_eval_chunks,
    )

    # ── Evaluate ─────────────────────────────────────────────────────
    run_quality = config.eval_context in ("quality", "both")
    run_controllability = config.eval_context in ("controllability", "both")

    quality_records = []
    ctrl_records = []
    total = len(val_df)

    for i, row in val_df.iterrows():
        dali_id = row["dali_id"]
        chunk_name = row["chunk_name"]
        print(f"\n[{timestamp()}] [{i + 1}/{total}] {dali_id}/{chunk_name}")

        try:
            arrays = _load_chunk_arrays(row, config)
        except Exception as e:
            print(f"  SKIP (load error): {e}")
            continue

        # RVC audio path (if provided)
        rvc_path = None
        if config.rvc_audio_dir:
            candidate = os.path.join(config.rvc_audio_dir, f"{dali_id}_{chunk_name}.wav")
            if os.path.exists(candidate):
                rvc_path = candidate

        # Context 1: Quality
        if run_quality:
            try:
                q = _eval_quality(arrays, model, config, device, speaker_emb, rvc_path, dali_id, chunk_name)
                q["dali_id"] = dali_id
                q["chunk_name"] = chunk_name
                quality_records.append(q)
                metric_summary = ", ".join(
                    f"{k}={v:.4f}" for k, v in q.items()
                    if isinstance(v, (int, float)) and k not in ("dali_id", "chunk_name")
                )
                print(f"  Quality: {metric_summary}")
            except Exception as e:
                print(f"  Quality FAILED: {e}")

        # Context 2: Controllability (DALI-path USTX priors)
        if run_controllability:
            rvc_dali_path = None
            if config.rvc_dali_audio_dir:
                candidate = os.path.join(config.rvc_dali_audio_dir, f"{dali_id}_{chunk_name}.wav")
                if os.path.exists(candidate):
                    rvc_dali_path = candidate

            try:
                c = _eval_controllability(
                    dali_id, chunk_name, arrays, model, config, device, speaker_emb,
                    rvc_dali_audio_path=rvc_dali_path,
                )
                if c is not None:
                    c["dali_id"] = dali_id
                    c["chunk_name"] = chunk_name
                    ctrl_records.append(c)
                    metric_summary = ", ".join(
                        f"{k}={v:.4f}" for k, v in c.items()
                        if isinstance(v, (int, float))
                    )
                    print(f"  Controllability: {metric_summary}")
            except Exception as e:
                print(f"  Controllability FAILED: {e}")

    # ── Aggregation ──────────────────────────────────────────────────
    if run_quality:
        q_df = _aggregate(quality_records, "Quality")
        if not q_df.empty:
            _save_results(q_df, config.output_dir, "quality")

    if run_controllability:
        c_df = _aggregate(ctrl_records, "Controllability")
        if not c_df.empty:
            _save_results(c_df, config.output_dir, "controllability")

    # ── Save config ──────────────────────────────────────────────────
    os.makedirs(config.output_dir, exist_ok=True)
    config.to_yaml(os.path.join(config.output_dir, "eval_config.yaml"))
    print(f"\n[{timestamp()}] Evaluation complete. Results in {config.output_dir}/")


if __name__ == "__main__":
    main()
