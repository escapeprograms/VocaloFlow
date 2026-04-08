import os
import sys
import re
import pythonnet
pythonnet.load("coreclr")
import clr

# Add DataSynthesizer root to path for cross-package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import UTAU_GENERATE_DLL

# Set up UtauGenerate import
dependency_dir = UTAU_GENERATE_DLL
if not os.path.exists(dependency_dir):
    print(f"Error: Could not find UtauGenerate at {dependency_dir}")
    sys.exit(1)

# Add the directory to PATH so native DLLs like onnxruntime.dll can be found by Pythonnet
bin_dir = os.path.dirname(dependency_dir)
native_dir = os.path.join(bin_dir, "runtimes", "win-x64", "native")
os.environ["PATH"] = bin_dir + os.pathsep + native_dir + os.pathsep + os.environ.get("PATH", "")

clr.AddReference(dependency_dir)
from UtauGenerate import Player

from utils.grab_midi import get_midi_pitch

# ---------------------------------------------------------------------------
# G2P engine — initialized once at module load
# ---------------------------------------------------------------------------
try:
    import g2p_en as _g2p_en_mod
    _G2P = _g2p_en_mod.G2p()
except Exception as _e:
    _G2P = None
    print(f"[WARNING] g2p_en not available: {_e}. use_phonemes will be disabled.")

_ARPABET_VOWELS = {'aa', 'ax', 'ae', 'ah', 'ao', 'aw', 'ay', 'eh', 'er', 'ey',
                   'ih', 'iy', 'ow', 'oy', 'uh', 'uw'}

def _normalize_arpabet(raw_phonemes):
    """Strip stress digits and lowercase: ['AH0','N','D'] -> ['ah','n','d']"""
    return [re.sub(r'[0-9]', '', p).lower() for p in raw_phonemes]

def _distribute_phonemes(phonemes, n_notes):
    """
    Coda-dominant syllabification: syllable i = phonemes[prev_end : next_vowel_pos].
    Final syllable = phonemes[last_vowel_pos : end].
    Returns a list of n_notes space-separated ARPAbet strings (may be empty for extras).
    """
    if not phonemes:
        return [''] * n_notes

    vowel_pos = [i for i, p in enumerate(phonemes) if p in _ARPABET_VOWELS]
    if not vowel_pos:
        # No vowels — put all consonants on first note, rest empty
        return [' '.join(phonemes)] + [''] * (n_notes - 1)

    syllables, prev = [], 0
    for si in range(len(vowel_pos)):
        end = vowel_pos[si + 1] if si < len(vowel_pos) - 1 else len(phonemes)
        syllables.append(' '.join(phonemes[prev:end]))
        prev = end

    if len(syllables) > n_notes:
        tail = ' '.join(p for s in syllables[n_notes - 1:] for p in s.split() if p)
        syllables = syllables[:n_notes - 1] + [tail]
    elif len(syllables) < n_notes:
        syllables += [''] * (n_notes - len(syllables))

    return syllables

def _build_text_fallback(is_continuation, word_text):
    """Intra-word legato is always applied: continuation notes get '+', first notes get the word text."""
    return "+" if is_continuation else word_text

def generate_prior_from_notes(chunk_dir, extracted_notes_path, player, use_phonemes=True):
    """Generate prior.wav and prior.ustx from ROSVOT-extracted notes.

    Reads extracted_notes.json (from ROSVOT note extraction with mapped lyrics)
    and generates prior audio via OpenUtau.

    Reuses the same OpenUtau tick conversion, phoneme distribution logic,
    and Player API as the original function.  No F0 pitch bends are applied
    (flat MIDI pitches from ROSVOT are used directly).

    Args:
        chunk_dir:            Directory containing extracted_notes.json; prior.wav
                              and prior.ustx will be written here.
        extracted_notes_path: Path to extracted_notes.json.
        player:               Pre-initialised OpenUtau Player instance.
        use_phonemes:         Use g2p_en ARPAbet phoneticHints per note.

    Returns:
        True if prior was generated, False otherwise.
    """
    import json
    import time

    with open(extracted_notes_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    notes = data.get("notes", [])
    if not notes:
        print(f"  No notes in {extracted_notes_path}, skipping prior generation.")
        return False

    player.resetParts()
    notes_added = 0

    # --- Pre-compute per-word ARPAbet phonemes ---
    # Group notes by word: identify word boundaries via note_type
    # (type 2 = first note of word, type 3 = continuation within word)
    word_groups = []  # list of (word_text, [note_indices])
    for i, note in enumerate(notes):
        if note["note_type"] == 2:
            word_groups.append((note["note_text"], [i]))
        elif note["note_type"] == 3 and word_groups:
            word_groups[-1][1].append(i)
        else:
            # Fallback: treat as new word
            word_groups.append((note["note_text"], [i]))

    word_phoneme_cache = {}   # word_group_idx -> list[str] normalised ARPAbet or None
    word_hint_cache = {}      # word_group_idx -> list[str] per-note hint strings

    if use_phonemes and _G2P is not None:
        for wg_idx, (word_text, _indices) in enumerate(word_groups):
            if not word_text or word_text == "-":
                word_phoneme_cache[wg_idx] = None
                continue
            try:
                word_phoneme_cache[wg_idx] = _normalize_arpabet(_G2P(word_text))
            except Exception as e:
                print(f"[WARNING] g2p_en failed for '{word_text}': {e}")
                word_phoneme_cache[wg_idx] = None

    # Build a reverse map: note_index -> (word_group_idx, position_within_word)
    note_to_word = {}
    for wg_idx, (word_text, indices) in enumerate(word_groups):
        for pos, ni in enumerate(indices):
            note_to_word[ni] = (wg_idx, pos)

    # --- Add notes to Player ---
    for i, note in enumerate(notes):
        start_s = note["start_s"]
        dur_s = note["note_dur"]
        midi_pitch = note["note_pitch"]
        note_type = note["note_type"]
        word_text = note["note_text"]
        is_continuation = (note_type == 3)

        # Determine display text with phoneme hints
        if use_phonemes and _G2P is not None and i in note_to_word:
            wg_idx, pos_in_word = note_to_word[i]
            wg_text, wg_indices = word_groups[wg_idx]
            phonemes_for_word = word_phoneme_cache.get(wg_idx)

            if phonemes_for_word is not None:
                # Compute hint distribution once per word
                if wg_idx not in word_hint_cache:
                    word_hint_cache[wg_idx] = _distribute_phonemes(
                        phonemes_for_word, len(wg_indices)
                    )
                hints = word_hint_cache[wg_idx]
                hint = hints[pos_in_word] if pos_in_word < len(hints) else ''
                if hint:
                    text = f"{wg_text}[{hint}]"
                else:
                    text = "+"
            else:
                text = _build_text_fallback(is_continuation, word_text if not is_continuation else wg_text)
        else:
            text = _build_text_fallback(is_continuation, word_text)

        position_ms = int(start_s * 1000)
        length_ms = int(dur_s * 1000)

        # Convert to OpenUtau ticks (120BPM, 480 res = 0.96 multiplier)
        position_ticks = max(0, int(position_ms * 0.96))
        length_ticks = max(15, int(length_ms * 0.96))

        player.addNote(position_ticks, length_ticks, midi_pitch, text)
        notes_added += 1

    if notes_added == 0:
        return False

    # Export wav + ustx under a single validate call
    segment_wav = os.path.join(chunk_dir, "prior.wav")
    segment_ustx = os.path.join(chunk_dir, "prior.ustx")
    os.makedirs(chunk_dir, exist_ok=True)
    player.export(segment_wav, segment_ustx)

    time.sleep(0.2)  # Race-condition buffer for OpenUtau C# thread on Array.Clear()

    print(f"  Prior generated: {notes_added} notes -> {segment_wav}")
    return True


def rerender_prior_with_adjusted_durations(chunk_dir, notes, player):
    """Fast re-render: clear and re-add notes with updated durations, skip phonemizer.

    Uses clearNotes() instead of resetParts() to preserve the part object and
    its cached phonemizer state. Renders via exportFast() which skips phonemizer
    re-setup. Must only be called after a full generate_prior_from_notes() has
    already run for this chunk (so phonemizer state is cached).

    Args:
        chunk_dir:  Directory for this chunk; prior.wav/prior.ustx will be overwritten.
        notes:      Note list with updated start_s / note_dur values.
        player:     Pre-initialised OpenUtau Player with cached phonemizer state.

    Returns:
        True if prior was re-rendered, False otherwise.
    """
    import time

    if not notes:
        return False

    player.clearNotes()
    for note in notes:
        position_ms = int(note["start_s"] * 1000)
        length_ms = int(note["note_dur"] * 1000)
        position_ticks = max(0, int(position_ms * 0.96))
        length_ticks = max(15, int(length_ms * 0.96))
        player.addNote(position_ticks, length_ticks, note["note_pitch"], note["note_text"])

    segment_wav = os.path.join(chunk_dir, "prior.wav")
    segment_ustx = os.path.join(chunk_dir, "prior.ustx")
    for f in [segment_wav, segment_ustx]:
        if os.path.exists(f):
            os.remove(f)
    player.exportFast(segment_wav, segment_ustx)

    time.sleep(0.2)  # Race-condition buffer for OpenUtau C# thread

    print(f"  Prior re-rendered (fast): {len(notes)} notes -> {segment_wav}")
    return True