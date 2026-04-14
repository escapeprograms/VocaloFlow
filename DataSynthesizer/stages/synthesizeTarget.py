import os
import sys
import json
import subprocess
import argparse
import re
from pathlib import Path

# Add DataSynthesizer root and repo root to path for cross-package imports
_DS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_DS_ROOT, ".."))
sys.path.insert(0, _DS_ROOT)
sys.path.insert(0, _REPO_ROOT)

from config import SOULX_PYTHON

# SOULX_DIR is used below to build subprocess PYTHONPATH / cwd and file paths
# into the SoulX-Singer source tree. No parent-process imports from SoulX-Singer
# are needed — the midi helpers are vendored in utils.midi_helpers.
SOULX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "SoulX-Singer"))

# Set up DALI import path
try:
    import DALI as dali_code
except ImportError:
    dali_code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DALI", "code"))
    sys.path.append(dali_code_path)
    try:
        import DALI as dali_code
    except ImportError:
        print("Error: Could not import DALI dataset library. Please ensure it is available.")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Extract DALI annotations to SoulX-Singer JSON and run inference.")
    parser.add_argument("--dali_id", default="006b5d1db6a447039c30443310b60c6f", help="DALI dataset entry ID")
    parser.add_argument("--language", default="English", help="Language for grapheme-to-phoneme (default: English)")
    parser.add_argument("--output_dir", default=os.path.abspath(os.path.dirname(__file__)), help="Output directory")
    parser.add_argument("--no_continuations", action="store_true", help="Disable merging of syllable note continuations")
    return parser.parse_args()

from utils.grab_midi import get_midi_pitch


def get_soulx_inference_config(provider: str = None) -> dict:
    """Return the SoulX-Singer model/config/phoneset paths + default prompt paths.

    Centralised here so both per-song (process_dali_to_target) and dataset-scale
    (synthesize_dataset.py) callers read from a single source of truth.

    The prompt_wav_path / prompt_metadata_path returned here are the provider's
    *first* prompt — used as CLI-level defaults by soulxsinger_batch_infer.py.
    Per-chunk prompt selection (for multi-prompt providers) overrides these via
    task-dict fields; see utils/prompt_selection.py.
    """
    from voice_providers import VOICE_PROVIDERS, DEFAULT_PROVIDER
    if provider is None:
        provider = DEFAULT_PROVIDER
    first_prompt = VOICE_PROVIDERS[provider]["prompts"][0]
    return {
        "model_path":          os.path.join(SOULX_DIR, "pretrained_models", "SoulX-Singer", "model.pt"),
        "config_path":         os.path.join(SOULX_DIR, "soulxsinger", "config", "soulxsinger.yaml"),
        "prompt_wav_path":     first_prompt["prompt_wav_path"],
        "prompt_metadata_path":first_prompt["prompt_metadata_path"],
        "phoneset_path":       os.path.join(SOULX_DIR, "soulxsinger", "utils", "phoneme", "phone_set.json"),
    }


def process_dali_to_target(dali_id="006b5d1db6a447039c30443310b60c6f", language="English", output_dir=None, use_continuations=True, mode="paragraph", n_lines=4, use_f0=False, save_mel=False, defer_inference=False, provider=None):
    if output_dir is None:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Data"))

    os.makedirs(output_dir, exist_ok=True)
    dali_data_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "DALI", "DALI_v2.0", "annot_tismir", f"{dali_id}.gz"))
    
    print(f"Loading DALI dataset entry from {dali_data_file} ...")
    try:
        target_entry = dali_code.get_an_entry(dali_data_file)
    except Exception as e:
        print(f"Error loading DALI entry: {e}")
        return
        
    if target_entry is None:
        print("Failed to load test song.")
        return
        
    print(f"Successfully loaded song: {target_entry.info['title']} by {target_entry.info['artist']}")

    try:
        target_entry.vertical2horizontal() 
    except Exception:
        pass # Already horizontal
    
    notes_annot = target_entry.annotations['annot'].get('notes', [])
    words_annot = target_entry.annotations['annot'].get('words', [])
    lines_annot = target_entry.annotations['annot'].get('lines', [])
    paragraphs_annot = target_entry.annotations['annot'].get('paragraphs', [])
    
    if not notes_annot or not words_annot or not lines_annot or not paragraphs_annot:
        print("Missing required annotations (notes, words, lines, or paragraphs).")
        return

    print(f"Generating SoulX-Singer annotations using mode: {mode}...")
    
    try:
        from utils.midi_helpers import notes2meta, Note
        
        from utils.determine_chunks import get_chunks
        chunks, chunk_start_times, chunk_names = get_chunks(mode, notes_annot, words_annot, lines_annot, paragraphs_annot, n_lines=n_lines)

        from utils.prompt_selection import select_prompt
        from voice_providers import DEFAULT_PROVIDER

        _provider = provider if provider is not None else DEFAULT_PROVIDER
        infer_cfg = get_soulx_inference_config(_provider)
        model_path           = infer_cfg["model_path"]
        config_path          = infer_cfg["config_path"]
        phoneset_path        = infer_cfg["phoneset_path"]

        inference_tasks = []

        for chunk_i, (note_indices, chunk_start_sec, chunk_name) in enumerate(zip(chunks, chunk_start_times, chunk_names)):
            
            # Setup output directory for this chunk
            chunk_out_dir = os.path.join(output_dir, dali_id, chunk_name)
            os.makedirs(chunk_out_dir, exist_ok=True)
            meta_path = os.path.join(chunk_out_dir, "music.json")
            
            soulx_notes = []
            note_indices_set = set(note_indices)
            for idx in note_indices:
                note_info = notes_annot[idx]
                word_idx = note_info['index']
                word_info = words_annot[word_idx]
                    
                word_text = word_info['text']
                
                # Intra-word continuation is always applied
                is_continuation = (idx > 0 and notes_annot[idx-1]['index'] == word_idx)

                if is_continuation:
                    text = "-"
                    note_type = 3
                else:
                    text = word_text
                    note_type = 2
                    
                start_time_sec = note_info['time'][0] - chunk_start_sec
                end_time_sec = note_info['time'][1] - chunk_start_sec
                dur_s = end_time_sec - start_time_sec
                
                freq_list = note_info['freq']
                midi_pitch = get_midi_pitch(freq_list)
                    
                if idx < len(notes_annot) - 1:
                    next_idx = idx + 1
                    # Only extend duration if the next note is within this chunk
                    if next_idx in note_indices_set:
                        next_note = notes_annot[next_idx]
                        next_start_time_sec = next_note['time'][0] - chunk_start_sec
                        # Always extend within a word; with use_continuations also extend across word boundaries
                        if next_note['index'] == word_idx or use_continuations:
                            dur_s = next_start_time_sec - start_time_sec
                        
                soulx_notes.append(
                    Note(
                        start_s=float(start_time_sec),
                        note_dur=float(dur_s),
                        note_text=str(text),
                        note_pitch=int(midi_pitch),
                        note_type=int(note_type)
                    )
                )
            
            if len(soulx_notes) > 0:
                # Select voice prompt based on chunk's MIDI content
                midi_pitches = [n.note_pitch for n in soulx_notes]
                prompt = select_prompt(_provider, midi_pitches)

                # Persist prompt selection for downstream phases
                prompt_info_path = os.path.join(chunk_out_dir, "prompt_info.json")
                with open(prompt_info_path, "w", encoding="utf-8") as f:
                    json.dump({
                        "provider": _provider,
                        "prompt_name": prompt["prompt_name"],
                    }, f, indent=2)

                notes2meta(
                    soulx_notes,
                    meta_path,
                    vocal_file=None, # Synthesizing, no original vocal provided
                    language=language,
                    pitch_extractor=None
                )

                if use_f0:
                    import numpy as np
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_list = json.load(f)
                    
                    time_res = 0.02
                    end_time_total = notes_annot[-1]['time'][1] + 10.0
                    try:
                        full_f0_vector = dali_code.annot2vector(notes_annot, end_time_total, time_res, type='melody')
                        
                        sum_dur = sum([float(x) for x in meta_list[0]['duration'].split()])
                        expected_frames = int(round(sum_dur / time_res))
                        
                        start_frame = int(chunk_start_sec / time_res)
                        chunk_f0 = full_f0_vector[start_frame : start_frame + expected_frames]
                        if len(chunk_f0) < expected_frames:
                            chunk_f0 = np.pad(chunk_f0, (0, expected_frames - len(chunk_f0)))
                        
                        meta_list[0]['f0'] = " ".join([str(round(f, 1)) for f in chunk_f0])
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta_list, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        print(f"Warning: Failed to extract F0 for SoulX-Singer: {e}")
                        use_f0 = False
                
                cmd_control = "melody" if use_f0 else "score"

                inference_tasks.append({
                    "target_metadata_path": meta_path,
                    "save_dir": chunk_out_dir,
                    "control": cmd_control,
                    "save_mel": save_mel,
                    "prompt_wav_path": prompt["prompt_wav_path"],
                    "prompt_metadata_path": prompt["prompt_metadata_path"],
                    "prompt_name": prompt["prompt_name"],
                })

        # Either return tasks for the caller to batch (defer_inference=True),
        # or launch the subprocess immediately for per-song use (defer_inference=False).
        if defer_inference:
            return inference_tasks

        if inference_tasks:
            import tempfile
            print(f"\nRunning SoulX-Singer batch inference for {len(inference_tasks)} chunk(s) (model loaded once)...")
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tf:
                json.dump(inference_tasks, tf)
                tasks_file = tf.name

            batch_script = os.path.join(os.path.dirname(__file__), "..", "batch", "soulxsinger_batch_infer.py")
            cmd = [
                SOULX_PYTHON,
                batch_script,
                "--tasks_json", tasks_file,
                "--model_path", model_path,
                "--config", config_path,
                "--prompt_wav_path", infer_cfg["prompt_wav_path"],
                "--prompt_metadata_path", infer_cfg["prompt_metadata_path"],
                "--phoneset_path", phoneset_path,
                "--device", "cuda",
                "--auto_shift",
                "--pitch_shift", "0",
            ]

            env = os.environ.copy()
            env["PYTHONPATH"] = SOULX_DIR + os.pathsep + env.get("PYTHONPATH", "")

            try:
                subprocess.run(cmd, env=env, cwd=SOULX_DIR, check=True)
                print(f"Batch inference complete for {len(inference_tasks)} chunk(s).")
            except subprocess.CalledProcessError as e:
                print(f"Batch inference failed with code {e.returncode}")
            finally:
                os.unlink(tasks_file)

    except Exception as e:
        print(f"Error extracting metadata from DALI annotations: {e}")
        return

if __name__ == "__main__":
    print("Running demo for DataSynthesizer module...")
    process_dali_to_target()

#conda run -n vocaloflow-datasynthesizer python synthesizePrior.py