"""DALI annotation → OpenUTAU USTX conversion for fair controllability evaluation.

Converts DALI MIDI annotations directly into USTX files + rendered prior WAVs,
bypassing SoulX-Singer so VocaloFlow gets an independent shot at following the score.

No pythonnet/CLR imports at module level — the OpenUTAU Player is only needed
inside :func:`generate_dali_ustx_for_chunk` and :func:`generate_all_dali_ustx`.
"""

from __future__ import annotations

import math
import os
import re
import time

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Phoneme helpers (copied from DataSynthesizer/stages/synthesizePrior.py
# because that module calls pythonnet.load("coreclr") at import time)
# ═══════════════════════════════════════════════════════════════════════════

_ARPABET_VOWELS = {
    "aa", "ax", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey",
    "ih", "iy", "ow", "oy", "uh", "uw",
}

_G2P = None


def _get_g2p():
    global _G2P
    if _G2P is None:
        try:
            import g2p_en
            _G2P = g2p_en.G2p()
        except Exception as e:
            print(f"[WARNING] g2p_en not available: {e}. Phoneme hints disabled.")
    return _G2P


def _normalize_arpabet(raw_phonemes: list[str]) -> list[str]:
    return [re.sub(r"[0-9]", "", p).lower() for p in raw_phonemes]


def _distribute_phonemes(phonemes: list[str], n_notes: int) -> list[str]:
    if not phonemes:
        return [""] * n_notes

    vowel_pos = [i for i, p in enumerate(phonemes) if p in _ARPABET_VOWELS]
    if not vowel_pos:
        return [" ".join(phonemes)] + [""] * (n_notes - 1)

    syllables, prev = [], 0
    for si in range(len(vowel_pos)):
        end = vowel_pos[si + 1] if si < len(vowel_pos) - 1 else len(phonemes)
        syllables.append(" ".join(phonemes[prev:end]))
        prev = end

    if len(syllables) > n_notes:
        tail = " ".join(p for s in syllables[n_notes - 1 :] for p in s.split() if p)
        syllables = syllables[: n_notes - 1] + [tail]
    elif len(syllables) < n_notes:
        syllables += [""] * (n_notes - len(syllables))

    return syllables


def _freq_to_midi(freq: float) -> int:
    if freq <= 0:
        return 0
    return int(round(69 + 12 * math.log2(freq / 440.0)))


# ═══════════════════════════════════════════════════════════════════════════
# DALI note extraction with word-level text
# ═══════════════════════════════════════════════════════════════════════════


def extract_dali_notes_with_words(
    dali_entry,
    chunk_start_s: float,
    chunk_end_s: float,
) -> list[dict]:
    """Extract DALI notes with word text and continuation status.

    Unlike :func:`~dali_ref.extract_dali_note_timings`, this preserves
    the note→word mapping from the DALI annotation hierarchy so we can
    build proper OpenUTAU lyrics (word starts vs. continuations).

    Returns list of dicts sorted by start time:
        start_s (chunk-relative), note_dur, note_text, note_pitch (MIDI),
        note_type (2=word start, 3=continuation)
    """
    notes_annot = dali_entry.annotations["annot"]["notes"]
    words_annot = dali_entry.annotations["annot"]["words"]

    in_range = []
    for idx, note in enumerate(notes_annot):
        note_start, note_end = note["time"]
        if note_end <= chunk_start_s or note_start >= chunk_end_s:
            continue
        in_range.append((idx, note))

    if not in_range:
        return []

    result = []
    prev_word_idx = None

    for idx, note in in_range:
        note_start, note_end = note["time"]
        word_idx = note["index"]
        word_text = words_annot[word_idx]["text"]

        is_continuation = prev_word_idx is not None and prev_word_idx == word_idx
        prev_word_idx = word_idx

        rel_start = max(0.0, note_start - chunk_start_s)
        rel_end = min(chunk_end_s - chunk_start_s, note_end - chunk_start_s)
        dur_s = rel_end - rel_start

        freq_hz = float(np.mean(note["freq"]))
        midi_pitch = _freq_to_midi(freq_hz)

        result.append({
            "start_s": rel_start,
            "note_dur": dur_s,
            "note_text": "-" if is_continuation else word_text,
            "note_pitch": midi_pitch,
            "note_type": 3 if is_continuation else 2,
        })

    # Extend durations to fill gaps (same pattern as synthesizeTarget.py:155-163)
    for i in range(len(result) - 1):
        next_start = result[i + 1]["start_s"]
        result[i]["note_dur"] = next_start - result[i]["start_s"]

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Format notes for OpenUTAU Player API
# ═══════════════════════════════════════════════════════════════════════════


def format_notes_for_player(
    notes: list[dict],
    use_phonemes: bool = True,
) -> list[tuple[int, int, int, str]]:
    """Convert note dicts into (position_ticks, length_ticks, midi_pitch, text).

    Applies g2p_en phoneme hints when *use_phonemes* is True, matching
    the approach in DataSynthesizer/stages/synthesizePrior.py.
    """
    g2p = _get_g2p() if use_phonemes else None

    # Group notes by word for phoneme distribution
    word_groups: list[tuple[str, list[int]]] = []
    for i, note in enumerate(notes):
        if note["note_type"] == 2:
            word_groups.append((note["note_text"], [i]))
        elif note["note_type"] == 3 and word_groups:
            word_groups[-1][1].append(i)
        else:
            word_groups.append((note["note_text"], [i]))

    # Pre-compute phonemes per word
    word_phonemes: dict[int, list[str] | None] = {}
    word_hints: dict[int, list[str]] = {}

    if g2p is not None:
        for wg_idx, (word_text, _indices) in enumerate(word_groups):
            if not word_text or word_text == "-":
                word_phonemes[wg_idx] = None
                continue
            try:
                word_phonemes[wg_idx] = _normalize_arpabet(g2p(word_text))
            except Exception:
                word_phonemes[wg_idx] = None

    # Reverse map: note index → (word_group_idx, position_within_word)
    note_to_word: dict[int, tuple[int, int]] = {}
    for wg_idx, (_text, indices) in enumerate(word_groups):
        for pos, ni in enumerate(indices):
            note_to_word[ni] = (wg_idx, pos)

    tuples = []
    for i, note in enumerate(notes):
        is_continuation = note["note_type"] == 3

        # Build display text with phoneme hints
        if g2p is not None and i in note_to_word:
            wg_idx, pos_in_word = note_to_word[i]
            wg_text, wg_indices = word_groups[wg_idx]
            phonemes = word_phonemes.get(wg_idx)

            if phonemes is not None:
                if wg_idx not in word_hints:
                    word_hints[wg_idx] = _distribute_phonemes(phonemes, len(wg_indices))
                hints = word_hints[wg_idx]
                hint = hints[pos_in_word] if pos_in_word < len(hints) else ""
                text = f"{wg_text}[{hint}]" if hint else "+"
            else:
                text = "+" if is_continuation else (note["note_text"] or "+")
        else:
            text = "+" if is_continuation else (note["note_text"] or "+")

        position_ms = int(note["start_s"] * 1000)
        length_ms = int(note["note_dur"] * 1000)
        position_ticks = max(0, int(position_ms * 0.96))
        length_ticks = max(15, int(length_ms * 0.96))

        tuples.append((position_ticks, length_ticks, note["note_pitch"], text))

    return tuples


# ═══════════════════════════════════════════════════════════════════════════
# USTX + WAV generation
# ═══════════════════════════════════════════════════════════════════════════


def generate_dali_ustx_for_chunk(
    dali_entry,
    chunk_name: str,
    output_dir: str,
    player,
    use_phonemes: bool = True,
) -> tuple[str, str] | None:
    """Generate USTX + WAV from DALI annotations for a single chunk.

    Args:
        dali_entry: Loaded DALI annotation entry (horizontal layout).
        chunk_name: e.g. ``'line_3'``.
        output_dir: Directory for this dali_id (e.g. ``dali_ustx/{dali_id}``).
        player: Initialised OpenUTAU Player instance.
        use_phonemes: Apply g2p phoneme hints.

    Returns:
        ``(ustx_path, wav_path)`` or ``None`` if no notes in this chunk.
    """
    from eval_utils.dali_ref import get_chunk_time_range

    try:
        chunk_start, chunk_end = get_chunk_time_range(dali_entry, chunk_name)
    except (IndexError, ValueError):
        return None

    notes = extract_dali_notes_with_words(dali_entry, chunk_start, chunk_end)
    if not notes:
        return None

    note_tuples = format_notes_for_player(notes, use_phonemes=use_phonemes)
    if not note_tuples:
        return None

    chunk_out = os.path.join(output_dir, chunk_name)
    os.makedirs(chunk_out, exist_ok=True)
    ustx_path = os.path.join(chunk_out, "dali_prior.ustx")
    wav_path = os.path.join(chunk_out, "dali_prior.wav")

    player.resetParts()
    for pos, dur, pitch, text in note_tuples:
        player.addNote(pos, dur, pitch, text)

    player.export(wav_path, ustx_path)
    time.sleep(0.2)  # race-condition buffer for OpenUtau C# thread

    return ustx_path, wav_path


def generate_all_dali_ustx(
    manifest_df,
    dali_annot_dir: str,
    output_base_dir: str,
    use_phonemes: bool = True,
) -> dict[tuple[str, str], tuple[str, str]]:
    """Generate DALI USTX files for all chunks in the validation manifest.

    Initialises the OpenUTAU Player once, processes every unique
    (dali_id, chunk_name) pair, and writes artifacts to disk.

    Returns:
        dict mapping ``(dali_id, chunk_name)`` → ``(ustx_path, wav_path)``.
    """
    from eval_utils.dali_ref import load_dali_entry

    # Late pythonnet import — only needed here
    import pythonnet
    pythonnet.load("coreclr")
    import clr

    import sys
    _repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, _repo_root)
    from config import UTAU_GENERATE_DLL

    bin_dir = os.path.dirname(UTAU_GENERATE_DLL)
    native_dir = os.path.join(bin_dir, "runtimes", "win-x64", "native")
    os.environ["PATH"] = (
        bin_dir + os.pathsep + native_dir + os.pathsep + os.environ.get("PATH", "")
    )
    clr.AddReference(UTAU_GENERATE_DLL)
    from UtauGenerate import Player

    player = Player("OpenUtau.Plugin.Builtin.ArpasingPlusPhonemizer")

    os.makedirs(output_base_dir, exist_ok=True)
    results: dict[tuple[str, str], tuple[str, str]] = {}
    success, skip, fail = 0, 0, 0

    seen = set()
    for _, row in manifest_df.iterrows():
        dali_id = row["dali_id"]
        chunk_name = row["chunk_name"]
        key = (dali_id, chunk_name)
        if key in seen:
            continue
        seen.add(key)

        dali_gz = os.path.join(dali_annot_dir, f"{dali_id}.gz")
        if not os.path.exists(dali_gz):
            skip += 1
            continue

        try:
            dali_entry = load_dali_entry(dali_annot_dir, dali_id)
        except Exception as e:
            print(f"  SKIP {dali_id}/{chunk_name}: load error: {e}")
            skip += 1
            continue

        song_dir = os.path.join(output_base_dir, dali_id)

        try:
            out = generate_dali_ustx_for_chunk(
                dali_entry, chunk_name, song_dir, player, use_phonemes
            )
            if out is None:
                skip += 1
            else:
                results[key] = out
                success += 1
                print(f"  [{success}] {dali_id}/{chunk_name}")
        except Exception as e:
            print(f"  FAIL {dali_id}/{chunk_name}: {e}")
            fail += 1

    print(f"\nGeneration complete: {success} ok, {skip} skipped, {fail} failed")
    return results
