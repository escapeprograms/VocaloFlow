# Utils Memory Palace

This directory contains **standalone utility modules** with no local cross-dependencies. These are leaf-level helpers used by stages and pipelines.

## File Details

### `vocoders.py`
Vocoder integration for converting mel-spectrograms back to audio waveforms.
- `mel_to_soulx_mel(mel, config)`: Converts a standard librosa mel-spectrogram to SoulX-Singer format (log-compressed, transposed).
- `SOULX_MEL_CONFIG`: SoulX-Singer mel parameters (sr=24000, n_fft=1920, hop=480, n_mels=128, fmin=0, fmax=12000, mel_mean=-4.92, mel_var=8.14).
- `invert_mel_to_audio(mel, config, vocoder)`: Dispatcher that routes to the appropriate vocoder.
- `invert_mel_to_audio_griffin_lim(mel, config)`: Griffin-Lim phase reconstruction (CPU, fast, lower quality).
- `invert_mel_to_audio_hifigan(mel, config)`: HiFIGAN neural vocoder (GPU, higher quality).
- `invert_mel_to_audio_soulxsinger(mel, config)`: SoulX-Singer's built-in Vocos vocoder (best quality for SoulX mel-spectrograms).
- Lazy-loads models on first call; caches globally.

### `grab_midi.py`
Frequency-to-MIDI conversion and per-note pitch computation from F0.
- `freq_to_midi(freq)`: Converts a raw Hz frequency into an integer MIDI note number.
- `get_midi_pitch(freqs)`: Returns the closest MIDI pitch for an average over a list of frequencies.
- `recompute_note_pitches(notes, f0, sr, hop)`: Recomputes `note_pitch` for each note from target F0 within its time window. Extracts F0 slice per note, computes mean voiced Hz, converts to MIDI via `freq_to_midi`. Also updates `mean_f0_hz` and `voiced_ratio` metadata. Falls back to existing `note_pitch` if no voiced frames found. Mutates notes in-place. Used by both `note_extraction_batch.py` (initial pitch) and `iterative_align.py` (pitch recalculation after duration adjustment).

### `determine_chunks.py`
Chunking utility for segmenting a song based on DALI annotations.
- `get_chunks(mode, notes, words, lines, paragraphs, n_lines)`: Takes the complete sets of DALI annotations and clusters notes into manageable processing segments.
- Supports grouping granularities: `test` (first line only), `line` (each line individually), `paragraph` (stanza by stanza), `n-line` (blocks of N lines).
- Returns `(chunks, chunk_start_times, chunk_names)`.

### `voiced_unvoiced.py`
Voiced/unvoiced extraction utilities supplementing F0 extraction. All frame-level defaults match SoulX-Singer config (sr=24000, hop=480).
- `get_voiced_mask(f0)`: Core primitive — boolean mask from F0 array (F0 > 0 = voiced). Single source of truth for voicing definition.
- `get_energy_mask(audio, sr, hop, threshold_db, frame_length)`: Boolean mask from RMS energy >= threshold. Uses librosa; hop-aligned to F0 grid.
- `get_validated_voiced_mask(f0, audio, ...)`: AND of F0 mask + energy mask for robust voicing. Truncates to min length to handle frame count mismatches.
- `compute_note_voicing_stats(f0, notes, sr, hop)`: Per-note statistics: `voiced_ratio`, `voiced_frames`, `total_frames`, `onset_frame`, `offset_frame`, `mean_f0_hz`. Used by `note_extraction_batch.py` to enrich `extracted_notes.json`.
- `find_unvoiced_gaps(f0, min_gap_frames, margin_frames, hop, sr)`: Detects contiguous unvoiced regions in the interior of an F0 array, ignoring edges. Generalizes the gap detection from `scratch/detect_silences.py`.
- `save_voicing(path, mask)` / `load_voicing(path)`: Save/load boolean mask as `.npy` (stored as `target_voicing.npy` alongside `target_f0.npy`).
- Standalone CLI: `python utils/voiced_unvoiced.py <target_f0.npy>` prints voicing summary stats.

### `phoneme_mask.py`
Generates frame-level phoneme identity mask for training, replicating SoulX-Singer's `DataProcessor.preprocess()` mel2note logic.
- `generate_phoneme_mask(chunk_dir, phoneset_path)`: Loads `adjusted_notes.json` (DTW-corrected durations) + `music.json` (phoneme strings from g2p). Builds expanded phoneme token sequence with `<BOW>`/`<EOW>` markers and English phoneme splitting. Saves two files:
  - `phoneme_ids.npy`: int32, shape `(P,)` — expanded phoneme token IDs (indices into `phone_set.json`)
  - `phoneme_mask.npy`: int32, shape `(T,)` — per-mel-frame index into `phoneme_ids` (same semantics as SoulX's `mel2note`)
- Falls back to `extracted_notes.json` if adjusted notes don't exist yet.
- Standalone CLI: `python utils/phoneme_mask.py --chunk_dir ../Data/<dali_id>/<chunk>`
- Frame grid matches mel: sr=24000, hop=480 (50fps).

### `generate_manifest.py`
Generates `manifest.csv` from iterative alignment results for ML training.
- `generate_manifest(data_dir, manifest_path)`: Walks `Data/{provider}/<dali_id>/<chunk_name>/` dirs, reads `alignment.json` (iterative format), writes CSV with columns: `provider`, `prompt_name`, `dali_id`, `chunk_name`, `prior_mel_path`, `target_mel_path`, `f0_path`, `voicing_path`, `phoneme_mask_path`, `adjusted_notes_path`, `converged`, `iterations`, `dtw_cost`, `mean_deviation`, `max_deviation`.
- `provider` and `prompt_name` are read from `alignment.json` (defaults to `"WillStetson"` for legacy data).
- Requires all training artifacts to exist before including a chunk in the manifest.
- Standalone CLI: `python utils/generate_manifest.py --data_dir ../Data/WillStetson`
- Uses `csv.DictWriter` (no pandas dependency). Idempotent — always overwrites.

### `prompt_selection.py`
Probabilistic voice prompt selection based on chunk MIDI content.
- `select_prompt(provider, midi_pitches)`: For multi-prompt providers (e.g. Rachie), computes median MIDI of the chunk, measures distance to each register's `midi_range`, applies softmax, and samples. Single-prompt providers return their only prompt.
- Returns dict with `prompt_name`, `prompt_wav_path`, `prompt_metadata_path`.
- No external dependencies — uses only `math`, `random`, `statistics` stdlib modules.
