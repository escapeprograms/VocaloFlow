"""Voice provider prompt configurations.

Each provider has a list of prompts. Single-prompt providers (e.g. WillStetson)
always use their one prompt. Multi-prompt providers (e.g. Rachie) have per-prompt
``midi_range`` tuples used by ``utils/prompt_selection.py`` for probabilistic
selection based on chunk MIDI content.

The ``prompt_name`` field is recorded per-chunk in prompt_info.json and
alignment.json so that the exact prompt used for synthesis is always traceable.
"""

import os

SOULX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "SoulX-Singer"))
VOCAL_PROMPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "VocalPrompts"))

VOICE_PROVIDERS = {
    "WillStetson": {
        "prompts": [
            {
                "prompt_name": "WillStetson",
                "prompt_wav_path": os.path.join(SOULX_DIR, "example", "transcriptions", "WillStetsonSample", "vocal.wav"),
                "prompt_metadata_path": os.path.join(SOULX_DIR, "example", "transcriptions", "WillStetsonSample", "metadata.json"),
            },
        ],
    },
    "Rachie": {
        "prompts": [
            {
                "prompt_name": "rachie_low",
                "midi_range": (55, 60),
                "prompt_wav_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_low", "vocal.wav"),
                "prompt_metadata_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_low", "metadata.json"),
            },
            {
                "prompt_name": "rachie_mid",
                "midi_range": (62, 67),
                "prompt_wav_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_mid", "vocal.wav"),
                "prompt_metadata_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_mid", "metadata.json"),
            },
            {
                "prompt_name": "rachie_high",
                "midi_range": (67, 74),
                "prompt_wav_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_high", "vocal.wav"),
                "prompt_metadata_path": os.path.join(VOCAL_PROMPTS_DIR, "Rachie", "rachie_high", "metadata.json"),
            },
        ],
    },
}

DEFAULT_PROVIDER = "Rachie"
