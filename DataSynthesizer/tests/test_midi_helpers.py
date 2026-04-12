"""
Unit tests for DataSynthesizer.utils.midi_helpers.

Covers:
- Note dataclass construction and end_s property
- notes2meta on the parent's call path (vocal_file=None, pitch_extractor=None)
- Byte-for-byte equivalence between vendored notes2meta and upstream
  SoulX-Singer/preprocess/tools/midi_parser.notes2meta for several scenarios.

These tests must pass BEFORE the synthesizeTarget.py import swap.
"""
import json
import os
import sys
from pathlib import Path

import pytest


# Make DataSynthesizer package importable when running pytest from repo root.
_HERE = Path(__file__).resolve().parent
_DS_ROOT = _HERE.parent
sys.path.insert(0, str(_DS_ROOT))

# Also add SoulX-Singer to path so we can import upstream for byte-equality tests.
_SOULX_DIR = _DS_ROOT.parent / "SoulX-Singer"
sys.path.insert(0, str(_SOULX_DIR))

from utils.midi_helpers import Note, notes2meta  # noqa: E402

# Upstream import (will pull torch via f0_extraction — that's fine under the fat env).
from preprocess.tools.midi_parser import (  # noqa: E402
    Note as UpstreamNote,
    notes2meta as upstream_notes2meta,
)


# ---------------------------------------------------------------------------
# Note dataclass
# ---------------------------------------------------------------------------

def test_note_construction_and_end_s():
    n = Note(start_s=1.25, note_dur=0.75, note_text="hello", note_pitch=62, note_type=2)
    assert n.start_s == 1.25
    assert n.note_dur == 0.75
    assert n.note_text == "hello"
    assert n.note_pitch == 62
    assert n.note_type == 2
    assert n.end_s == pytest.approx(2.0)


def test_note_end_s_with_zero_duration():
    n = Note(start_s=3.0, note_dur=0.0, note_text="<SP>", note_pitch=0, note_type=1)
    assert n.end_s == 3.0


# ---------------------------------------------------------------------------
# Fixtures: realistic note lists mirroring what synthesizeTarget builds
# ---------------------------------------------------------------------------

def _english_phrase_notes():
    """A short English phrase — exercises the g2p_en path without any Chinese chars."""
    return [
        Note(start_s=0.0, note_dur=0.40, note_text="hello", note_pitch=64, note_type=2),
        Note(start_s=0.40, note_dur=0.30, note_text="-",    note_pitch=64, note_type=3),
        Note(start_s=0.70, note_dur=0.35, note_text="world", note_pitch=67, note_type=2),
        Note(start_s=1.05, note_dur=0.25, note_text="-",    note_pitch=67, note_type=3),
    ]


def _phrase_with_long_gap():
    """Two phrases separated by a > MAX_GAP_SEC silence → should split into two segments."""
    return [
        Note(start_s=0.0,  note_dur=0.40, note_text="sing",   note_pitch=64, note_type=2),
        Note(start_s=0.40, note_dur=0.30, note_text="-",      note_pitch=64, note_type=3),
        # long gap via an explicit <SP> note lasting > 2.0s
        Note(start_s=0.70, note_dur=2.50, note_text="<SP>",   note_pitch=0,  note_type=1),
        Note(start_s=3.20, note_dur=0.35, note_text="again",  note_pitch=67, note_type=2),
        Note(start_s=3.55, note_dur=0.25, note_text="-",      note_pitch=67, note_type=3),
    ]


def _long_phrase_forcing_duration_split():
    """A phrase whose duration sum exceeds MAX_SEGMENT_DUR_SUM_SEC (60s) → forces split."""
    notes = []
    t = 0.0
    # 80 notes at 1.0s each = 80s total, should trigger the duration cut
    for i in range(80):
        notes.append(
            Note(
                start_s=t,
                note_dur=1.0,
                note_text="word" if i % 2 == 0 else "-",
                note_pitch=60 + (i % 12),
                note_type=2 if i % 2 == 0 else 3,
            )
        )
        t += 1.0
    return notes


def _empty_text_notes():
    """Notes with empty text strings — should be coerced into <SP> by notes2meta."""
    return [
        Note(start_s=0.0,  note_dur=0.30, note_text="",      note_pitch=0,  note_type=1),
        Note(start_s=0.30, note_dur=0.40, note_text="hello", note_pitch=62, note_type=2),
        Note(start_s=0.70, note_dur=0.20, note_text="-",     note_pitch=62, note_type=3),
    ]


ALL_SCENARIOS = {
    "english_phrase":       _english_phrase_notes,
    "phrase_with_long_gap": _phrase_with_long_gap,
    "long_phrase_split":    _long_phrase_forcing_duration_split,
    "empty_text_notes":     _empty_text_notes,
}


# ---------------------------------------------------------------------------
# Shape / key assertions on parent's call path
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_name", list(ALL_SCENARIOS.keys()))
def test_notes2meta_output_shape(tmp_path, scenario_name):
    notes = ALL_SCENARIOS[scenario_name]()
    out_path = tmp_path / "music.json"

    notes2meta(
        notes,
        str(out_path),
        vocal_file=None,
        language="English",
        pitch_extractor=None,
    )

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1

    required_keys = {
        "index", "language", "time", "duration", "text",
        "phoneme", "note_pitch", "note_type", "f0",
    }
    for seg in data:
        assert required_keys.issubset(seg.keys()), f"missing keys in {seg}"
        # time is [start_ms, end_ms]
        assert isinstance(seg["time"], list) and len(seg["time"]) == 2
        assert seg["language"] == "English"
        # f0 is empty string when pitch_extractor is None
        assert seg["f0"] == ""
        # duration / text / note_pitch / note_type are space-joined strings
        assert isinstance(seg["duration"], str)
        assert isinstance(seg["text"], str)
        assert isinstance(seg["note_pitch"], str)
        assert isinstance(seg["note_type"], str)


def test_long_phrase_splits_into_multiple_segments(tmp_path):
    notes = _long_phrase_forcing_duration_split()
    out_path = tmp_path / "music.json"
    notes2meta(notes, str(out_path), vocal_file=None, language="English", pitch_extractor=None)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(data) >= 2, "Expected multiple segments after duration-sum split"


def test_long_gap_splits_into_multiple_segments(tmp_path):
    notes = _phrase_with_long_gap()
    out_path = tmp_path / "music.json"
    notes2meta(notes, str(out_path), vocal_file=None, language="English", pitch_extractor=None)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(data) == 2, f"Expected 2 segments, got {len(data)}"


# ---------------------------------------------------------------------------
# Byte-equality vs upstream (the strongest correctness check)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("scenario_name", list(ALL_SCENARIOS.keys()))
def test_byte_equality_with_upstream(tmp_path, scenario_name):
    """
    Run vendored and upstream notes2meta on identical inputs and diff the JSON.
    This is the strongest correctness check — any serialization drift fails here.
    """
    vendored_notes = ALL_SCENARIOS[scenario_name]()
    upstream_notes = [
        UpstreamNote(
            start_s=n.start_s,
            note_dur=n.note_dur,
            note_text=n.note_text,
            note_pitch=n.note_pitch,
            note_type=n.note_type,
        )
        for n in vendored_notes
    ]

    # The "index" field inside notes2meta output is derived from the basename of
    # the meta_path. To make byte-equality meaningful, both runs must write to a
    # file with the SAME basename — so we use two subdirectories.
    vendored_dir = tmp_path / "vendored"
    upstream_dir = tmp_path / "upstream"
    vendored_dir.mkdir()
    upstream_dir.mkdir()
    vendored_out = vendored_dir / "music.json"
    upstream_out = upstream_dir / "music.json"

    notes2meta(
        vendored_notes,
        str(vendored_out),
        vocal_file=None,
        language="English",
        pitch_extractor=None,
    )
    upstream_notes2meta(
        upstream_notes,
        str(upstream_out),
        vocal_file=None,
        language="English",
        pitch_extractor=None,
    )

    vendored_bytes = vendored_out.read_bytes()
    upstream_bytes = upstream_out.read_bytes()

    if vendored_bytes != upstream_bytes:
        vendored_json = json.loads(vendored_bytes)
        upstream_json = json.loads(upstream_bytes)
        pytest.fail(
            f"Byte mismatch for scenario '{scenario_name}'.\n"
            f"vendored: {json.dumps(vendored_json, ensure_ascii=False, indent=2)}\n"
            f"upstream: {json.dumps(upstream_json, ensure_ascii=False, indent=2)}"
        )
