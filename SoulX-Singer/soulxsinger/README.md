# SoulXSinger Memory Palace

## Architecture Overview

SoulXSinger is a zero-shot singing voice synthesis model. Its core pipeline involves:
1. **Encoders**: Text (phoneme), pitch (MIDI score), and f0 (melody) encoders.
2. **Pre-flow**: A sequence of ConvNeXtV2 blocks.
3. **CFM Decoder**: A Conditional Flow Matching decoder that generates the mel-spectrogram.
4. **Vocoder**: A Vocos-based vocoder that converts the mel-spectrogram to audio.

## Core Module: `soulxsinger/models/soulxsinger.py`

### `SoulXSinger` Class

- `__init__(self, config)`: Initializes the encoders, pre-flow blocks, CFM decoder, and vocoder.
- `infer(self, meta, auto_shift=False, pitch_shift=0, n_steps=32, cfg=3, control="melody", return_mel=False)`:
    - Performs inference using either `melody` or `score` control.
    - Handles optional pitch shifting.
    - Uses `cfm_decoder.reverse_diffusion` to generate the mel-spectrogram.
    - Passes the mel-spectrogram through the `vocoder`.
    - **Added**: `return_mel` flag. If `True`, returns `(generated_audio, generated_mel)`.

## CLI: `cli/inference.py`

- `process(args, config, model)`:
    - Runs the full inference pipeline for multiple segments.
    - Merges the generated audio segments.
    - **Added**: `--save_mel` flag support. Accumulates `generated_mel` segments and saves them as `generated_mel.npy`.

## Vocoder: `soulxsinger/models/modules/vocoder.py`

- `Vocoder` class: A wrapper around the Vocos base model.
- `load_vocos_model`: Loads the Vocos architecture and weights.
- `Vocos`: The main vocoder architecture with a backbone and an ISTFT head.

## Decoder: `soulxsinger/models/modules/decoder.py`

- `CFMDecoder`: Manages the flow matching process.
- `reverse_diffusion`: The core inference method for generating the mel-spectrogram.
