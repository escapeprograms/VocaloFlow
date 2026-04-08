# Memory Palace: SoulX-Singer/preprocess/tools/

## midi_parser.py

Central module for converting between SoulX-Singer metadata JSON and MIDI files. Uses a `Note` dataclass (`start_s`, `note_dur`, `note_text`, `note_pitch`, `note_type`) as the intermediate representation.

### Constants

- `SAMPLE_RATE = 44100` — audio sample rate for wav cuts
- `MAX_GAP_SEC = 2.0` — silence gap threshold to split segments
- `MAX_SEGMENT_DUR_SUM_SEC = 60.0` — max accumulated note duration before forcing a segment split
- `END_EXTENSION_SEC = 0.4` — pad each segment end with silence for model context
- `SILENCE_THRESHOLD_SEC = 0.2` — threshold for inserting explicit `<SP>` between notes in `midi2notes`

### Key Functions

- **`_seconds_to_ticks(seconds, ticks_per_beat, tempo)`** — Converts seconds to MIDI ticks.

- **`_append_segment_to_meta(...)`** — Helper that takes accumulated note lists (`note_text`, `note_dur`, etc.) and appends a single segment dict to `meta_data`. Handles optional wav cutting (`cut_wavs_output_dir`), F0 extraction, and G2P phoneme conversion. If `vocal_file` is provided, it cuts audio and adds an `END_EXTENSION_SEC` trailing `<SP>` pad. The `time` field is set to `[start_ms, end_ms]` based on the first and last note boundaries.

- **`notes2meta(notes, meta_path, vocal_file, language, pitch_extractor)`** — Main entry point for converting a list of `Note` objects into a SoulX-Singer metadata JSON file. Iterates notes and accumulates them into segments, flushing when:
  1. A trailing `<SP>` exceeds `MAX_GAP_SEC` (the `<SP>` is stripped before flushing).
  2. Adding the next note would push `dur_sum` past `MAX_SEGMENT_DUR_SUM_SEC`.
  3. End of notes — trailing `<SP>` notes are stripped before the final flush; if only `<SP>` remains, the segment is skipped entirely.

  The `append_note` inner function merges consecutive `<SP>` notes into one.

- **`meta2notes(meta_path)`** — Parses a SoulX-Singer metadata JSON back into a flat list of `Note` with absolute `start_s` times. Reconstructs timing from `time[0]` offset + cumulative durations.

- **`notes2midi(notes, midi_path)`** — Writes a list of `Note` to a standard MIDI file. Uses `note_type == 3` to mark continuations (lyric set to `"-"`). Includes tempo and time signature meta events.

- **`midi2notes(midi_path)`** — Parses a MIDI file into a list of `Note`. Handles lyric events, inserts `<SP>` for silences exceeding `SILENCE_THRESHOLD_SEC`, and maps MIDI note events to `Note` objects.

- **`midi2meta(midi_path, meta_path, vocal_file, language)`** — Convenience wrapper: `midi2notes` -> `notes2meta`.

### Segmentation Behavior

The segmentation in `notes2meta` is critical for inference performance. Each segment becomes a separate inference call in SoulX-Singer. Segments that are too long (especially pure-silence segments) can cause the model to hang or OOM during flow-matching diffusion. The mid-loop gap check strips trailing `<SP>` before flushing; the final flush also strips trailing `<SP>` to prevent generating unnecessary silence.

## g2p.py

Grapheme-to-phoneme conversion via `g2p_transform(note_text_list, language)`. Returns a list of phoneme strings for each note text. Used by `_append_segment_to_meta` to populate the `phoneme` field in metadata.

## f0_extraction.py

Provides `F0Extractor` class for extracting fundamental frequency from audio. Used optionally by `_append_segment_to_meta` when a `pitch_extractor` is provided.
