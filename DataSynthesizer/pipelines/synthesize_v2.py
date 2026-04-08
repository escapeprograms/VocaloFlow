"""
Target-first synthesis pipeline — per-song driver.

Generates Target audio first (SoulX-Singer), extracts notes from it via ROSVOT,
then generates Prior audio (OpenUtau) from those extracted notes.  Finally runs
DTW alignment.  This is the v2 counterpart of synthesize.py.

Usage:
    conda run -n data_synthesizer python synthesize_v2.py --dali_id <ID> --mode line
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Imports from existing modules
# ---------------------------------------------------------------------------
from stages.synthesizeTarget import process_dali_to_target
from utils.determine_chunks import get_chunks

# DALI import
try:
    import DALI as dali_code
except ImportError:
    dali_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DALI", "code"))
    sys.path.append(dali_code_path)
    import DALI as dali_code

SOULX_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))

from config import SOULX_PYTHON
DEFAULT_OUTPUT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_chunk_words(note_indices, notes_annot, words_annot):
    """Return the ordered unique word texts for a chunk's note indices.

    Used to provide DALI lyrics for ROSVOT lyric mapping.
    """
    seen = set()
    words = []
    for idx in note_indices:
        wid = notes_annot[idx]['index']
        if wid not in seen:
            seen.add(wid)
            words.append(words_annot[wid]['text'])
    return words


def save_chunk_words(dali_id, output_dir, mode, n_lines):
    """Load DALI entry, compute chunks, and save chunk_words.json per chunk.

    Returns list of (chunk_name, chunk_dir) tuples.
    """
    dali_data_file = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "DALI", "DALI_v2.0", "annot_tismir", f"{dali_id}.gz"
    ))
    target_entry = dali_code.get_an_entry(dali_data_file)
    if target_entry is None:
        print(f"Failed to load DALI entry {dali_id}")
        return []

    try:
        target_entry.vertical2horizontal()
    except Exception:
        pass

    notes_annot = target_entry.annotations['annot'].get('notes', [])
    words_annot = target_entry.annotations['annot'].get('words', [])
    lines_annot = target_entry.annotations['annot'].get('lines', [])
    paragraphs_annot = target_entry.annotations['annot'].get('paragraphs', [])

    if not notes_annot or not words_annot or not lines_annot or not paragraphs_annot:
        print("Missing required DALI annotations.")
        return []

    chunks, _, chunk_names = get_chunks(mode, notes_annot, words_annot, lines_annot, paragraphs_annot, n_lines=n_lines)

    result = []
    for note_indices, chunk_name in zip(chunks, chunk_names):
        chunk_dir = os.path.join(output_dir, dali_id, chunk_name)
        os.makedirs(chunk_dir, exist_ok=True)

        words = extract_chunk_words(note_indices, notes_annot, words_annot)
        words_path = os.path.join(chunk_dir, "chunk_words.json")
        with open(words_path, "w", encoding="utf-8") as f:
            json.dump(words, f, ensure_ascii=False)

        result.append((chunk_name, chunk_dir))

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def synthesize_v2(
    dali_id="006b5d1db6a447039c30443310b60c6f",
    output_dir=None,
    mode="line",
    n_lines=4,
    use_f0=False,
    use_continuations=True,
    use_phonemes=True,
    provider=None,
):
    from voice_providers import DEFAULT_PROVIDER
    if provider is None:
        provider = DEFAULT_PROVIDER
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT

    # All data lives under Data/{provider}/
    provider_dir = os.path.join(output_dir, provider)
    os.makedirs(provider_dir, exist_ok=True)

    ts = lambda: datetime.now().strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    # Phase 1: Generate music.json + chunk_words.json from DALI
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === Phase 1: Target Metadata ===")
    chunk_info = save_chunk_words(dali_id, provider_dir, mode, n_lines)
    print(f"  Saved chunk_words.json for {len(chunk_info)} chunks.")

    # ------------------------------------------------------------------
    # Phase 2: SoulX-Singer inference (inline, not batched for single song)
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === Phase 2: SoulX-Singer Inference ===")
    process_dali_to_target(
        dali_id=dali_id,
        output_dir=provider_dir,
        use_continuations=use_continuations,
        mode=mode,
        n_lines=n_lines,
        use_f0=use_f0,
        save_mel=True,
        defer_inference=False,  # run inline for single song
        provider=provider,
    )
    print(f"[{ts()}] Target synthesis complete.")

    # ------------------------------------------------------------------
    # Phase 3: Note extraction + F0 (subprocess in soulxsinger env)
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === Phase 3: Note Extraction + F0 ===")

    # Build extraction tasks from chunks that have target.wav
    extraction_tasks = []
    for chunk_name, chunk_dir in chunk_info:
        audio_path = os.path.join(chunk_dir, "target.wav")
        notes_path = os.path.join(chunk_dir, "extracted_notes.json")
        f0_path = os.path.join(chunk_dir, "target_f0.npy")
        words_path = os.path.join(chunk_dir, "chunk_words.json")

        if not os.path.exists(audio_path):
            continue
        if os.path.exists(notes_path) and os.path.exists(f0_path):
            print(f"  Skipping {chunk_name}: already extracted")
            continue

        words = []
        if os.path.exists(words_path):
            with open(words_path, "r", encoding="utf-8") as f:
                words = json.load(f)

        extraction_tasks.append({
            "chunk_dir": chunk_dir,
            "audio_path": audio_path,
            "item_name": chunk_name,
            "words": words,
        })

    if extraction_tasks:
        rmvpe_base = os.path.join(SOULX_DIR, "pretrained_models", "SoulX-Singer-Preprocess")
        batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "note_extraction_batch.py")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tf:
            json.dump(extraction_tasks, tf)
            tasks_file = tf.name

        cmd = [
            SOULX_PYTHON, batch_script,
            "--tasks_json",         tasks_file,
            "--rmvpe_model_path",   os.path.join(rmvpe_base, "rmvpe", "rmvpe.pt"),
            "--device",             "cuda",
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = SOULX_DIR + os.pathsep + env.get("PYTHONPATH", "")

        try:
            subprocess.run(cmd, env=env, cwd=SOULX_DIR, check=True)
            print(f"  Note extraction complete ({len(extraction_tasks)} chunks).")
        except subprocess.CalledProcessError as e:
            print(f"  Note extraction FAILED with return code {e.returncode}.")
        finally:
            os.unlink(tasks_file)
    else:
        print("  No chunks to extract (all done or no target.wav).")

    # ------------------------------------------------------------------
    # Phase 4: Generate priors + iterative alignment
    # ------------------------------------------------------------------
    print(f"\n[{ts()}] === Phase 4: Prior Generation + Iterative Alignment ===")
    from stages.synthesizePrior import Player
    from alignment.iterative_align import iterative_align
    from utils.phoneme_mask import generate_phoneme_mask

    player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")

    for chunk_name, chunk_dir in chunk_info:
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
            print(f"  Skipping {chunk_name}: alignment.json exists")
            continue

        # Load notes and lyrics text
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

        print(f"  [{ts()}] Iterative alignment for {chunk_name}...")
        adjusted_notes, metrics = iterative_align(
            chunk_dir=chunk_dir,
            notes=notes,
            target_audio_path=target_audio,
            lyrics_text=lyrics_text,
            player=player,
            use_phonemes=use_phonemes,
            max_iterations=3,
            duration_threshold=0.15,
        )

        # Save the final adjusted notes
        adjusted_path = os.path.join(chunk_dir, "adjusted_notes.json")
        with open(adjusted_path, "w", encoding="utf-8") as f:
            json.dump({"notes": adjusted_notes, "source": "iterative"}, f, indent=2)

        # Save alignment metrics with per-chunk provider tracking
        prompt_info_path = os.path.join(chunk_dir, "prompt_info.json")
        if os.path.exists(prompt_info_path):
            with open(prompt_info_path, "r", encoding="utf-8") as f:
                prompt_info = json.load(f)
            metrics["provider"] = prompt_info["provider"]
            metrics["prompt_name"] = prompt_info["prompt_name"]
        else:
            metrics["provider"] = provider
            metrics["prompt_name"] = provider
        with open(alignment_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        # Generate phoneme identity mask from adjusted notes + music.json
        generate_phoneme_mask(chunk_dir)

    print(f"\n[{ts()}] === v2 Pipeline Complete ===")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Target-first synthesis pipeline (single song)."
    )
    parser.add_argument("--dali_id", default="006b5d1db6a447039c30443310b60c6f")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=["line", "n-line", "paragraph", "test"], default="line")
    parser.add_argument("--n_lines", type=int, default=4)
    parser.add_argument("--use_f0", action="store_true")
    parser.add_argument("--use_continuations", action="store_true", default=True)
    parser.add_argument("--use_phonemes", action="store_true", default=True)
    parser.add_argument("--provider", default=None,
                        help="Voice provider name (default: WillStetson).")

    args = parser.parse_args()
    synthesize_v2(
        dali_id=args.dali_id,
        output_dir=args.output_dir,
        mode=args.mode,
        n_lines=args.n_lines,
        use_f0=args.use_f0,
        use_continuations=args.use_continuations,
        use_phonemes=args.use_phonemes,
        provider=args.provider,
    )
