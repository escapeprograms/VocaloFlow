"""Probabilistic voice prompt selection based on chunk MIDI content.

For providers with multiple prompts (e.g. Rachie with low/mid/high registers),
selects a prompt using softmax sampling over the distance from the chunk's
median MIDI pitch to each register's range.

Single-prompt providers always return their only prompt.
"""

import math
import random
from statistics import median
from typing import List

from voice_providers import VOICE_PROVIDERS


def select_prompt(provider: str, midi_pitches: List[int]) -> dict:
    """Select a voice prompt for a chunk based on its MIDI pitch content.

    Args:
        provider:      Provider name (key in VOICE_PROVIDERS).
        midi_pitches:  List of MIDI note numbers for the chunk's notes.

    Returns:
        Dict with keys: prompt_name, prompt_wav_path, prompt_metadata_path.
    """
    prompts = VOICE_PROVIDERS[provider]["prompts"]

    # Single prompt — no selection needed
    if len(prompts) == 1:
        return prompts[0]

    # Multi-prompt — softmax over register distance
    med = median(midi_pitches)
    scores = []
    for p in prompts:
        lo, hi = p["midi_range"]
        distance = max(0, lo - med, med - hi)
        scores.append(-distance)

    # Softmax
    max_score = max(scores)
    exps = [math.exp(s - max_score) for s in scores]
    total = sum(exps)
    probs = [e / total for e in exps]

    # Sample
    r = random.random()
    cumulative = 0.0
    for i, prob in enumerate(probs):
        cumulative += prob
        if r < cumulative:
            return prompts[i]

    # Fallback (floating point edge case)
    return prompts[-1]
