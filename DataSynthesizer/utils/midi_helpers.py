"""
Vendored SoulX-Singer metadata writer (Note dataclass + notes2meta + g2p_transform).

This module is a self-contained copy of the parent-process subset of
SoulX-Singer/preprocess/tools/midi_parser.py and g2p.py. It exists so that
DataSynthesizer's pipeline can import Note/notes2meta without triggering
midi_parser.py's `from .f0_extraction import F0Extractor` line, which pulls
torch into the parent process even though F0Extractor is never instantiated
on the parent's call path (vocal_file=None, pitch_extractor=None).

Byte-equivalence with upstream notes2meta is covered by
DataSynthesizer/tests/test_midi_helpers.py. Keep this file in sync with
upstream if the upstream serialization format changes.
"""
import json
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, List

import librosa
from soundfile import write

import ToJyutping
from g2pM import G2pM
from g2p_en import G2p as G2pE


# ---------------------------------------------------------------------------
# g2p (vendored from SoulX-Singer/preprocess/tools/g2p.py)
# ---------------------------------------------------------------------------

_EN_WORD_RE = re.compile(r"^[A-Za-z]+(?:'[A-Za-z]+)*$")
_ZH_WORD_RE = re.compile(r"[\u4e00-\u9fff]")

EN_FLAG = "en_"
YUE_FLAG = "yue_"
ZH_FLAG = "zh_"

g2p_zh = G2pM()
g2p_en = G2pE()


def is_chinese_char(word: str) -> bool:
    if len(word) != 1:
        return False
    return bool(_ZH_WORD_RE.fullmatch(word))


def is_english_word(word: str) -> bool:
    if not word:
        return False
    return bool(_EN_WORD_RE.fullmatch(word))


def g2p_cantonese(sent):
    return ToJyutping.get_jyutping_list(sent)       # with tone


def g2p_mandarin(sent):
    return g2p_zh(sent, tone=True, char_split=False)


def g2p_english(word):
    return g2p_en(word)


def g2p_transform(words, lang):

    zh_words = []
    transformed_words = [0] * len(words)

    for idx, w in enumerate(words):
        if w == "<SP>":
            transformed_words[idx] = w
            continue

        w = w.replace("?", "").replace(".", "").replace("!", "").replace(",", "")

        if is_chinese_char(w):
            zh_words.append([idx, w])
        else:
            if is_english_word(w):
                w = EN_FLAG + "-".join(g2p_english(w.lower()))
            else:
                w = "<SP>"
        transformed_words[idx] = w

    sent = "".join([k[1] for k in zh_words])

    # zh (zh and yue) transformer to g2p
    if len(sent) > 0:
        if lang == "Cantonese":
            g2pm_rst = g2p_cantonese(sent)       # with tone
            g2pm_rst = [YUE_FLAG + k[1] for k in g2pm_rst]
        else:
            g2pm_rst = g2p_mandarin(sent)
            g2pm_rst = [ZH_FLAG + k for k in g2pm_rst]
        for p, w in zip([k[0] for k in zh_words], g2pm_rst):
            transformed_words[p] = w

    return transformed_words


# ---------------------------------------------------------------------------
# Note / notes2meta (vendored from SoulX-Singer/preprocess/tools/midi_parser.py)
# ---------------------------------------------------------------------------

# Audio and segmentation constants
SAMPLE_RATE = 44100             # Audio sample rate for any wav cuts during midi2meta
END_EXTENSION_SEC = 0.4         # Extend each segment end by this much silence (sec) to give the model more context
MAX_GAP_SEC = 2.0               # Gap threshold to split segments in midi2meta (sec)
MAX_SEGMENT_DUR_SUM_SEC = 60.0  # Max total duration sum of notes in a single metadata segment before splitting into multiple segments (sec)


@dataclass
class Note:
    """Single note: text, duration (seconds), pitch (MIDI), type. start_s is absolute start time in seconds (for ordering / MIDI)."""
    start_s: float
    note_dur: float
    note_text: str
    note_pitch: int
    note_type: int

    @property
    def end_s(self) -> float:
        return self.start_s + self.note_dur


def _append_segment_to_meta(
    meta_data: List[dict],
    meta_path_str: str,
    cut_wavs_output_dir: str | None,
    vocal_file: str | None,
    language: str,
    audio_data: Any | None,
    pitch_extractor: Any | None,
    note_start: List[float],
    note_end: List[float],
    note_text: List[Any],
    note_pitch: List[Any],
    note_type: List[Any],
    note_dur: List[float],
) -> None:
    """Helper function for midi2meta to append the current segment (accumulated in note_*) to meta_data list, with optional wav cut and pitch extraction."""
    if not all((note_start, note_end, note_text, note_pitch, note_type, note_dur)):
        return

    base_name = os.path.splitext(os.path.basename(meta_path_str))[0]
    item_name = f"{base_name}_{len(meta_data)}"
    wav_fn = None
    if cut_wavs_output_dir and vocal_file and audio_data is not None:
        wav_fn = os.path.join(cut_wavs_output_dir, f"{item_name}.wav")
        end_pad = int(END_EXTENSION_SEC * SAMPLE_RATE)
        start_sample = max(0, int(note_start[0] * SAMPLE_RATE))
        end_sample = min(len(audio_data), int(note_end[-1] * SAMPLE_RATE) + end_pad)

        end_pad_dur = (end_sample / SAMPLE_RATE - note_end[-1]) if end_sample > int(note_end[-1] * SAMPLE_RATE) else 0.0
        if end_pad_dur > 0:
            note_dur = note_dur + [end_pad_dur]
            note_text = note_text + ["<SP>"]
            note_pitch = note_pitch + [0]
            note_type = note_type + [1]
        start_ms = int(start_sample / SAMPLE_RATE * 1000)
        end_ms = int(end_sample / SAMPLE_RATE * 1000)
        write(wav_fn, audio_data[start_sample:end_sample], SAMPLE_RATE)
    else:
        start_ms = int(note_start[0] * 1000)
        end_ms = int(note_end[-1] * 1000)

    if pitch_extractor is not None:
        if not wav_fn or not os.path.isfile(wav_fn):
            raise FileNotFoundError(f"Segment wav file not found: {wav_fn}")
        f0 = pitch_extractor.process(wav_fn)
    else:
        f0 = []

    note_text_list = list(note_text)
    note_pitch_list = list(note_pitch)
    note_type_list = list(note_type)
    note_dur_list = list(note_dur)

    meta_data.append(
        {
            "index": item_name,
            "language": language,
            "time": [start_ms, end_ms],
            "duration": " ".join(str(round(x, 2)) for x in note_dur_list),
            "text": " ".join(note_text_list),
            "phoneme": " ".join(g2p_transform(note_text_list, language)),
            "note_pitch": " ".join(str(x) for x in note_pitch_list),
            "note_type": " ".join(str(x) for x in note_type_list),
            "f0": " ".join(str(round(float(x), 1)) for x in f0),
        }
    )


def notes2meta(
    notes: List[Note],
    meta_path: str,
    vocal_file: str | None,
    language: str,
    pitch_extractor: Any | None,
) -> None:
    """Write SoulX-Singer metadata JSON from a list of Note (segmenting + wav cuts)."""
    meta_path_str = str(meta_path)

    cut_wavs_output_dir = None
    if vocal_file:
        cut_wavs_output_dir = os.path.join(os.path.dirname(vocal_file), "cut_wavs_tmp")
        os.makedirs(cut_wavs_output_dir, exist_ok=True)

    note_text: List[Any] = []
    note_pitch: List[Any] = []
    note_type: List[Any] = []
    note_dur: List[float] = []
    note_start: List[float] = []
    note_end: List[float] = []
    meta_data: List[dict] = []
    audio_data = None
    if vocal_file:
        audio_data, _ = librosa.load(vocal_file, sr=SAMPLE_RATE, mono=True)
    dur_sum = 0.0

    def flush_current_segment() -> None:
        nonlocal dur_sum
        _append_segment_to_meta(
            meta_data,
            meta_path_str,
            cut_wavs_output_dir,
            vocal_file,
            language,
            audio_data,
            pitch_extractor,
            note_start,
            note_end,
            note_text,
            note_pitch,
            note_type,
            note_dur,
        )
        note_text.clear()
        note_pitch.clear()
        note_type.clear()
        note_dur.clear()
        note_start.clear()
        note_end.clear()
        dur_sum = 0.0

    def append_note(start: float, end: float, text: str, pitch: int, type_: int) -> None:
        nonlocal dur_sum
        duration = end - start
        if duration <= 0:
            return

        if len(note_text) > 0 and text == "<SP>" and note_text[-1] == "<SP>":
            note_dur[-1] += duration
            note_end[-1] = end
        else:
            note_text.append(text)
            note_pitch.append(pitch)
            note_type.append(type_)
            note_dur.append(duration)
            note_start.append(start)
            note_end.append(end)
        dur_sum += duration

    for note in notes:
        start = float(note.start_s)
        end = float(note.end_s)
        text = note.note_text
        pitch = note.note_pitch
        type_ = note.note_type

        if text == "" or pitch == "" or type_ == "":
            append_note(start, end, "<SP>", 0, 1)
            continue

        # cut the segment when ends with a long <SP> note
        if (
            len(note_text) > 0
            and note_text[-1] == "<SP>"
            and note_dur[-1] > MAX_GAP_SEC
        ):
            note_text.pop()
            note_pitch.pop()
            note_type.pop()
            note_dur.pop()
            note_start.pop()
            note_end.pop()

            dur_sum = sum(note_dur)
            flush_current_segment()

        # cut the segment if adding the current note would exceed the max duration sum threshold
        if dur_sum + (end - start) > MAX_SEGMENT_DUR_SUM_SEC and len(note_text) > 0:
            flush_current_segment()

        append_note(start, end, text, int(pitch), int(type_))

    if note_text:
        flush_current_segment()

    with open(meta_path_str, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, ensure_ascii=False, indent=2)

    if cut_wavs_output_dir:
        try:
            shutil.rmtree(cut_wavs_output_dir, ignore_errors=True)
        except Exception:
            pass
