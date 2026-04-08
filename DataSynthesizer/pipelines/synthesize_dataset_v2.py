"""
Dataset-scale target-first synthesis pipeline (v2).

Flips the generation order: Target audio is generated first from DALI annotations,
then ROSVOT extracts note timings/pitches from the target audio, and Prior audio
is generated from those extracted notes.  This produces priors that match what
SoulX-Singer actually sang, improving alignment quality.

Five-phase pipeline that amortises all expensive model loads:

  Phase 1 — Target Metadata      : Generate music.json + chunk_words.json from DALI.
  Phase 2 — Inference             : Run SoulX-Singer once per batch of songs.
  Phase 3 — Extraction            : Run ROSVOT + F0 once per batch of songs.
  Phase 4 — Iterative Alignment   : Generate prior via OpenUtau and iteratively
                                    adjust note durations until DTW convergence.
  Phase 5 — Manifest              : Generate manifest.csv for ML training.

All five phases are independently resumable via sentinel files per chunk.

Usage:
    python synthesize_dataset_v2.py --phases 12345 --songs_per_batch 100
    python synthesize_dataset_v2.py --phases 34    --songs_per_batch 50
    python synthesize_dataset_v2.py --dali_ids <id1> <id2>
"""

import argparse
import gzip
import json
import os
import pickle
import subprocess
import sys
import tempfile
from datetime import datetime
from typing import List, Optional

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stages.synthesizeTarget import process_dali_to_target, get_soulx_inference_config
from pipelines.synthesize_v2 import save_chunk_words
from config import SOULX_PYTHON

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOULX_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))
DALI_ANNOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DALI", "DALI_v2.0", "annot_tismir"))
DEFAULT_OUTPUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))

# File written at end of Phase 1 so Phase 2 can be re-run independently
TASKS_CACHE_NAME = "pending_inference_tasks.json"


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def get_english_dali_ids(dali_annot_dir: str) -> List[str]:
    """Scan the DALI annotation directory and return IDs for all English songs."""
    ids = []
    for fname in sorted(os.listdir(dali_annot_dir)):
        if not fname.endswith(".gz"):
            continue
        try:
            with gzip.open(os.path.join(dali_annot_dir, fname), "rb") as fh:
                obj = pickle.load(fh)
            lang = obj.info.get("metadata", {}).get("language", "")
            if lang.lower() == "english":
                ids.append(fname.replace(".gz", ""))
        except Exception:
            pass  # Corrupt or unreadable entry — skip silently
    return ids


# ---------------------------------------------------------------------------
# Inference subprocess launcher
# ---------------------------------------------------------------------------

def _launch_inference_subprocess(tasks: List[dict], infer_cfg: dict) -> None:
    """Write tasks to a temp JSON file and launch soulxsinger_batch_infer.py."""
    batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "soulxsinger_batch_infer.py")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(tasks, tf)
        tasks_file = tf.name

    cmd = [
        SOULX_PYTHON, batch_script,
        "--tasks_json",            tasks_file,
        "--model_path",            infer_cfg["model_path"],
        "--config",                infer_cfg["config_path"],
        "--prompt_wav_path",       infer_cfg["prompt_wav_path"],
        "--prompt_metadata_path",  infer_cfg["prompt_metadata_path"],
        "--phoneset_path",         infer_cfg["phoneset_path"],
        "--device",                "cuda",
        "--auto_shift",
        "--pitch_shift",           "0",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SOULX_DIR + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, env=env, cwd=SOULX_DIR, check=True)
        print(f"  Batch complete ({len(tasks)} tasks).")
    except subprocess.CalledProcessError as e:
        print(f"  Batch FAILED with return code {e.returncode}.")
    finally:
        os.unlink(tasks_file)


# ---------------------------------------------------------------------------
# Phase 1 — Target Metadata
# ---------------------------------------------------------------------------

def run_phase1_target_metadata(
    dali_ids: List[str],
    output_dir: str,
    mode: str,
    n_lines: int,
    use_f0: bool,
    use_continuations: bool,
    save_mel: bool,
    provider: str = None,
) -> List[dict]:
    """Generate music.json + chunk_words.json for every song; collect inference tasks.

    Returns the combined list of all SoulX-Singer inference task dicts.
    """
    all_tasks: List[dict] = []
    total = len(dali_ids)

    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 1 | {i+1}/{total}] {dali_id}")
        try:
            # Save chunk_words.json for each chunk (needed by Phase 3)
            save_chunk_words(dali_id, output_dir, mode, n_lines)

            # Generate music.json and collect inference tasks
            tasks = process_dali_to_target(
                dali_id=dali_id,
                output_dir=output_dir,
                mode=mode,
                n_lines=n_lines,
                use_f0=use_f0,
                use_continuations=use_continuations,
                save_mel=save_mel,
                defer_inference=True,
                provider=provider,
            )
            if tasks:
                all_tasks.extend(tasks)
                print(f"  -> {len(tasks)} chunk(s) queued.")
        except Exception as e:
            print(f"  [Phase 1] ERROR for {dali_id}: {e}")

    return all_tasks


# ---------------------------------------------------------------------------
# Phase 2 — Inference
# ---------------------------------------------------------------------------

def run_phase2_inference(all_tasks: List[dict], songs_per_batch: int, provider: str = None) -> None:
    """Split task list into song-sized batches and run one subprocess per batch.

    The SoulX-Singer model is loaded once per subprocess.
    """
    if not all_tasks:
        print("[Phase 2] No tasks to run.")
        return

    infer_cfg = get_soulx_inference_config(provider)

    def dali_id_from_task(task: dict) -> str:
        return os.path.basename(os.path.dirname(task["save_dir"]))

    seen: dict = {}
    for task in all_tasks:
        did = dali_id_from_task(task)
        seen.setdefault(did, []).append(task)

    song_ids = list(seen.keys())
    total_batches = max(1, (len(song_ids) + songs_per_batch - 1) // songs_per_batch)

    print(f"\n[Phase 2] {len(all_tasks)} tasks across {len(song_ids)} songs "
          f"-> {total_batches} batch(es) of up to {songs_per_batch} songs each.")

    for batch_i in range(total_batches):
        batch_song_ids = song_ids[batch_i * songs_per_batch : (batch_i + 1) * songs_per_batch]
        batch_tasks = [task for sid in batch_song_ids for task in seen[sid]]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 2 | Batch {batch_i+1}/{total_batches}] "
              f"{len(batch_song_ids)} songs, {len(batch_tasks)} chunks...")
        _launch_inference_subprocess(batch_tasks, infer_cfg)


# ---------------------------------------------------------------------------
# Phase 3 — Note Extraction + F0
# ---------------------------------------------------------------------------

def collect_extraction_tasks(
    dali_ids: List[str],
    output_dir: str,
) -> List[dict]:
    """Scan output dirs for chunks with target.wav but no extracted_notes.json."""
    tasks = []
    for dali_id in dali_ids:
        dali_dir = os.path.join(output_dir, dali_id)
        if not os.path.isdir(dali_dir):
            continue
        for chunk_name in sorted(os.listdir(dali_dir)):
            chunk_dir = os.path.join(dali_dir, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            audio_path = os.path.join(chunk_dir, "target.wav")
            notes_path = os.path.join(chunk_dir, "extracted_notes.json")
            f0_path = os.path.join(chunk_dir, "target_f0.npy")

            if not os.path.exists(audio_path):
                continue
            if os.path.exists(notes_path) and os.path.exists(f0_path):
                continue

            # Load chunk words for lyric mapping
            words = []
            words_path = os.path.join(chunk_dir, "chunk_words.json")
            if os.path.exists(words_path):
                with open(words_path, "r", encoding="utf-8") as f:
                    words = json.load(f)

            tasks.append({
                "chunk_dir": chunk_dir,
                "audio_path": audio_path,
                "item_name": chunk_name,
                "words": words,
            })
    return tasks


def _launch_extraction_subprocess(tasks: List[dict]) -> None:
    """Write tasks to a temp JSON file and launch note_extraction_batch.py."""
    batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "note_extraction_batch.py")
    rmvpe_base = os.path.join(SOULX_DIR, "pretrained_models", "SoulX-Singer-Preprocess")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False, encoding="utf-8"
    ) as tf:
        json.dump(tasks, tf)
        tasks_file = tf.name

    cmd = [
        SOULX_PYTHON, batch_script,
        "--tasks_json",        tasks_file,
        "--rmvpe_model_path",  os.path.join(rmvpe_base, "rmvpe", "rmvpe.pt"),
        "--device",            "cuda",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = SOULX_DIR + os.pathsep + env.get("PYTHONPATH", "")

    try:
        subprocess.run(cmd, env=env, cwd=SOULX_DIR, check=True)
        print(f"  Batch complete ({len(tasks)} tasks).")
    except subprocess.CalledProcessError as e:
        print(f"  Batch FAILED with return code {e.returncode}.")
    finally:
        os.unlink(tasks_file)


def run_phase3_extraction(
    all_tasks: List[dict],
    songs_per_batch: int,
) -> None:
    """Split extraction tasks into batches and run one subprocess per batch.

    ROSVOT + RMVPE models are loaded once per subprocess.
    """
    if not all_tasks:
        print("[Phase 3] No tasks to run.")
        return

    # Group by dali_id for batching
    def dali_id_from_task(task: dict) -> str:
        # chunk_dir is <output_dir>/<dali_id>/<chunk_name>
        return os.path.basename(os.path.dirname(task["chunk_dir"]))

    seen: dict = {}
    for task in all_tasks:
        did = dali_id_from_task(task)
        seen.setdefault(did, []).append(task)

    song_ids = list(seen.keys())
    total_batches = max(1, (len(song_ids) + songs_per_batch - 1) // songs_per_batch)

    print(f"\n[Phase 3] {len(all_tasks)} tasks across {len(song_ids)} songs "
          f"-> {total_batches} batch(es) of up to {songs_per_batch} songs each.")

    for batch_i in range(total_batches):
        batch_song_ids = song_ids[batch_i * songs_per_batch : (batch_i + 1) * songs_per_batch]
        batch_tasks = [task for sid in batch_song_ids for task in seen[sid]]
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 3 | Batch {batch_i+1}/{total_batches}] "
              f"{len(batch_song_ids)} songs, {len(batch_tasks)} chunks...")
        _launch_extraction_subprocess(batch_tasks)


# ---------------------------------------------------------------------------
# Phase 4 — Iterative Alignment
# ---------------------------------------------------------------------------

def run_phase4_iterative_alignment(
    dali_ids: List[str],
    output_dir: str,
    use_phonemes: bool = True,
    max_iterations: int = 3,
    duration_threshold: float = 0.15,
    player=None,
    provider: str = None,
) -> None:
    """Generate prior and iteratively align it to target for all chunks.

    Initialises OpenUtau Player once (if not provided).  For each chunk, runs
    iterative_align which generates the prior via OpenUtau and adjusts note
    durations until DTW convergence (or max_iterations reached).

    Resume sentinel: alignment.json — chunks that already have it are skipped.
    """
    from stages.synthesizePrior import Player
    from alignment.iterative_align import iterative_align
    from utils.phoneme_mask import generate_phoneme_mask

    if player is None:
        player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")
    total = len(dali_ids)

    for i, dali_id in enumerate(dali_ids):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] [Phase 4 | {i+1}/{total}] {dali_id}")
        dali_dir = os.path.join(output_dir, dali_id)
        if not os.path.isdir(dali_dir):
            continue

        for chunk_name in sorted(os.listdir(dali_dir)):
            chunk_dir = os.path.join(dali_dir, chunk_name)
            if not os.path.isdir(chunk_dir):
                continue

            notes_path = os.path.join(chunk_dir, "extracted_notes.json")
            target_audio = os.path.join(chunk_dir, "target.wav")
            words_path = os.path.join(chunk_dir, "chunk_words.json")
            alignment_path = os.path.join(chunk_dir, "alignment.json")

            if not os.path.exists(notes_path) or not os.path.exists(target_audio):
                continue
            if os.path.exists(alignment_path):
                # Alignment done — but ensure phoneme mask exists (crash recovery)
                phoneme_mask_path = os.path.join(chunk_dir, "phoneme_mask.npy")
                if not os.path.exists(phoneme_mask_path):
                    generate_phoneme_mask(chunk_dir)
                continue

            with open(notes_path, "r", encoding="utf-8") as f:
                notes_data = json.load(f)
            notes = notes_data.get("notes", [])
            if not notes:
                continue

            words = []
            if os.path.exists(words_path):
                with open(words_path, "r", encoding="utf-8") as f:
                    words = json.load(f)
            lyrics_text = " ".join(words) if words else ""

            try:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"  [{ts}] Iterative alignment for {chunk_name}...")
                adjusted_notes, metrics = iterative_align(
                    chunk_dir=chunk_dir,
                    notes=notes,
                    target_audio_path=target_audio,
                    lyrics_text=lyrics_text,
                    player=player,
                    use_phonemes=use_phonemes,
                    max_iterations=max_iterations,
                    duration_threshold=duration_threshold,
                )

                adjusted_path = os.path.join(chunk_dir, "adjusted_notes.json")
                with open(adjusted_path, "w", encoding="utf-8") as f:
                    json.dump({"notes": adjusted_notes, "source": "iterative"}, f, indent=2)

                # Tag metrics with per-chunk voice provider info for traceability
                prompt_info_path = os.path.join(chunk_dir, "prompt_info.json")
                if os.path.exists(prompt_info_path):
                    with open(prompt_info_path, "r", encoding="utf-8") as f:
                        prompt_info = json.load(f)
                    metrics["provider"] = prompt_info["provider"]
                    metrics["prompt_name"] = prompt_info["prompt_name"]
                else:
                    # Legacy data without prompt_info.json
                    metrics["provider"] = provider or "WillStetson"
                    metrics["prompt_name"] = provider or "WillStetson"

                with open(alignment_path, "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                # Generate phoneme identity mask from adjusted notes + music.json
                generate_phoneme_mask(chunk_dir)

            except Exception as e:
                print(f"  [Phase 4] ERROR for {chunk_name}: {e}")


# ---------------------------------------------------------------------------
# Phase 5 — Manifest Generation
# ---------------------------------------------------------------------------

def run_phase5_manifest(output_dir: str) -> None:
    """Generate manifest.csv from iterative alignment results."""
    from utils.generate_manifest import generate_manifest

    path = generate_manifest(output_dir)
    # Count rows (subtract 1 for header)
    with open(path, "r", encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    print(f"  Manifest written to {path} ({n_rows} chunks).")


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def synthesize_dataset_v2(
    output_dir: str = DEFAULT_OUTPUT,
    dali_annot_dir: str = DALI_ANNOT_DIR,
    mode: str = "line",
    n_lines: int = 4,
    use_f0: bool = False,
    use_continuations: bool = True,
    use_phonemes: bool = True,
    max_iterations: int = 3,
    duration_threshold: float = 0.15,
    songs_per_batch: int = 100,
    phases: str = "12345",
    dali_ids: Optional[List[str]] = None,
    provider: str = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
) -> None:
    """Run the full target-first dataset synthesis pipeline.

    Args:
        output_dir:           Root Data directory (provider subdir is appended).
        dali_annot_dir:       Path to DALI annot_tismir directory.
        mode:                 Chunk granularity — 'line', 'n-line', 'paragraph', 'test'.
        n_lines:              Lines per chunk when mode='n-line'.
        use_f0:               Use F0 curves for SoulX-Singer conditioning.
        use_continuations:    Extend note durations to fill intra-word gaps.
        use_phonemes:         Use ARPAbet phoneticHints via g2p_en.
        max_iterations:       Max iterative alignment iterations per chunk.
        duration_threshold:   Per-note ratio tolerance for convergence (e.g. 0.15 = 15%).
        songs_per_batch:      Songs grouped per subprocess call.
        phases:               Which phases to run, e.g. '12345', '34', '5'.
        dali_ids:             If given, process only these IDs; otherwise all English.
        provider:             Voice provider name (default: WillStetson).
    """
    from voice_providers import DEFAULT_PROVIDER
    if provider is None:
        provider = DEFAULT_PROVIDER

    # All data lives under Data/{provider}/
    provider_dir = os.path.join(output_dir, provider)
    inference_cache = os.path.join(provider_dir, TASKS_CACHE_NAME)
    os.makedirs(provider_dir, exist_ok=True)

    # Resolve song list
    if dali_ids is None:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Scanning for English DALI entries...")
        dali_ids = get_english_dali_ids(dali_annot_dir)
        print(f"Found {len(dali_ids)} English entries.")

    # Apply index slice for multi-machine partitioning
    total_before_slice = len(dali_ids)
    dali_ids = dali_ids[start_index:end_index]
    print(f"Processing songs[{start_index}:{end_index}] — "
          f"{len(dali_ids)} of {total_before_slice} songs")

    _sep = "=" * 62
    _batch_sep = "#" * 62
    has_phases_1234 = any(p in phases for p in "1234")

    # --- Pre-initialise expensive resources once ---
    player = None
    if "4" in phases:
        from stages.synthesizePrior import Player
        player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")

    # --- Batch loop (Phases 1-4) ---
    if has_phases_1234:
        total_batches = max(1, (len(dali_ids) + songs_per_batch - 1) // songs_per_batch)

        for batch_i in range(total_batches):
            batch_start = batch_i * songs_per_batch
            batch_end = min(batch_start + songs_per_batch, len(dali_ids))
            batch_ids = dali_ids[batch_start:batch_end]

            print(f"\n{_batch_sep}")
            print(f"  BATCH {batch_i+1}/{total_batches}  "
                  f"({len(batch_ids)} songs: indices {batch_start}–{batch_end-1})")
            print(_batch_sep)

            # ---- Phase 1 for this batch ----
            batch_tasks: List[dict] = []
            if "1" in phases:
                print(f"\n{_sep}")
                print(f"  PHASE 1 — Target Metadata  ({len(batch_ids)} songs)")
                print(_sep)
                batch_tasks = run_phase1_target_metadata(
                    batch_ids, provider_dir, mode, n_lines,
                    use_f0, use_continuations, save_mel=True,
                    provider=provider,
                )
                # Merge into cumulative cache (replace entries for this batch's songs)
                existing_cache: List[dict] = []
                if os.path.exists(inference_cache):
                    with open(inference_cache, "r", encoding="utf-8") as f:
                        existing_cache = json.load(f)
                batch_id_set = set(batch_ids)
                existing_cache = [
                    t for t in existing_cache
                    if os.path.basename(os.path.dirname(t["save_dir"])) not in batch_id_set
                ]
                existing_cache.extend(batch_tasks)
                with open(inference_cache, "w", encoding="utf-8") as f:
                    json.dump(existing_cache, f)
                print(f"\n[Phase 1] Batch {batch_i+1} complete — "
                      f"{len(batch_tasks)} inference tasks.")

            # ---- Phase 2 for this batch ----
            if "2" in phases:
                print(f"\n{_sep}")
                print(f"  PHASE 2 — SoulX-Singer Inference")
                print(_sep)
                if "1" not in phases:
                    # Load from cache and filter to this batch's songs
                    if not os.path.exists(inference_cache):
                        raise FileNotFoundError(
                            f"Inference task cache not found at {inference_cache}. "
                            f"Run Phase 1 first."
                        )
                    with open(inference_cache, "r", encoding="utf-8") as f:
                        all_cached = json.load(f)
                    batch_id_set = set(batch_ids)
                    batch_tasks = [
                        t for t in all_cached
                        if os.path.basename(os.path.dirname(t["save_dir"])) in batch_id_set
                    ]
                run_phase2_inference(batch_tasks, songs_per_batch=songs_per_batch, provider=provider)

            # ---- Phase 3 for this batch ----
            if "3" in phases:
                print(f"\n{_sep}")
                print(f"  PHASE 3 — Note Extraction + F0")
                print(_sep)
                extraction_tasks = collect_extraction_tasks(batch_ids, provider_dir)
                print(f"  {len(extraction_tasks)} extraction tasks collected.")
                run_phase3_extraction(extraction_tasks, songs_per_batch=songs_per_batch)

            # ---- Phase 4 for this batch ----
            if "4" in phases:
                print(f"\n{_sep}")
                print(f"  PHASE 4 — Iterative Alignment  ({len(batch_ids)} songs)")
                print(_sep)
                run_phase4_iterative_alignment(
                    batch_ids, provider_dir,
                    use_phonemes=use_phonemes,
                    max_iterations=max_iterations,
                    duration_threshold=duration_threshold,
                    player=player,
                    provider=provider,
                )

    # --- Phase 5 — once after all batches ---
    if "5" in phases:
        print(f"\n{_sep}")
        print(f"  PHASE 5 — Manifest Generation")
        print(_sep)
        run_phase5_manifest(provider_dir)

    print(f"\n{_sep}")
    print(f"  v2 Dataset synthesis complete.")
    print(_sep)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Target-first dataset synthesis pipeline (Phase 1-5)."
    )
    parser.add_argument("--output_dir",          default=DEFAULT_OUTPUT)
    parser.add_argument("--dali_annot_dir",      default=DALI_ANNOT_DIR)
    parser.add_argument("--mode",                default="line",
                        choices=["line", "n-line", "paragraph", "test"])
    parser.add_argument("--n_lines",             type=int, default=4)
    parser.add_argument("--use_f0",              action="store_true")
    parser.add_argument("--use_continuations",   action="store_true", default=True)
    parser.add_argument("--use_phonemes",        action="store_true", default=True)
    parser.add_argument("--max_iterations",      type=int, default=3,
                        help="Max iterative alignment iterations per chunk (default: 3).")
    parser.add_argument("--duration_threshold",  type=float, default=0.15,
                        help="Per-note ratio tolerance for convergence (default: 0.15).")
    parser.add_argument("--songs_per_batch",     type=int, default=100,
                        help="Songs grouped per subprocess (default: 100).")
    parser.add_argument("--phases",              default="12345",
                        help="Which phases to run: 1=metadata, 2=inference, "
                             "3=extraction, 4=iterative alignment, 5=manifest.")
    parser.add_argument("--dali_ids",            nargs="*", default=None,
                        help="Optional subset of DALI IDs to process.")
    parser.add_argument("--provider",            default=None,
                        help="Voice provider name (default: Rachie).")
    parser.add_argument("--start_index",         type=int, default=0,
                        help="First song index, inclusive (default: 0).")
    parser.add_argument("--end_index",           type=int, default=None,
                        help="Last song index, exclusive (default: end of list).")

    args = parser.parse_args()
    args.use_continuations = True
    args.mode = "line"
    args.use_phonemes = True

    synthesize_dataset_v2(
        output_dir=args.output_dir,
        dali_annot_dir=args.dali_annot_dir,
        mode=args.mode,
        n_lines=args.n_lines,
        use_f0=args.use_f0,
        use_continuations=args.use_continuations,
        use_phonemes=args.use_phonemes,
        max_iterations=args.max_iterations,
        duration_threshold=args.duration_threshold,
        songs_per_batch=args.songs_per_batch,
        phases=args.phases,
        dali_ids=args.dali_ids,
        provider=args.provider,
        start_index=args.start_index,
        end_index=args.end_index,
    )
