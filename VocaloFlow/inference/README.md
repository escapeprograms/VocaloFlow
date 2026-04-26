# VocaloFlow Inference — Memory Palace

End-to-end inference for VocaloFlow: takes an OpenUTAU `.ustx` (plus a pre-rendered prior WAV) and produces a high-quality singing voice WAV by running a learned flow-matching ODE between the OpenUTAU prior mel and the target mel, then vocoding via SoulX-Singer's Vocos.

## Files

### `pipeline.py` — End-to-end CLI
Top-level orchestration. Run as `python -m inference.pipeline ...` from the `VocaloFlow/` directory.

**Path setup (lines 28–47):** Inserts `VocaloFlow/` and `SoulX-Singer/` into `sys.path` but **deliberately excludes** `DataSynthesizer/` because its `utils/` package would shadow VocaloFlow's own `utils/`. DataSynthesizer modules (`vocoders.py`, `phoneme_mask.py`, `voiced_unvoiced.py`) are loaded via `importlib.util.spec_from_file_location` instead.

**Constants:** `SR = 24000`, `HOP = 480` (i.e. ~50 fps, 1 frame ≈ 20 ms).

**Functions:**
- `parse_ustx(ustx_path)` — Loads the `.ustx` YAML, pulls `bpm`, `resolution`, computes `ms_per_tick`, and returns the first voice part's notes (`position_ticks`, `duration_ticks`, `tone`, `lyric`, `pitch_points`, `snap_first`, `vibrato`). Also extracts part-level `pitd_curve` (pitch deviation curve with `xs`/`ys` arrays, usually None).
- `render_ustx_to_wav(ustx_path, notes_data, output_wav)` — Optional fallback that calls the C# `UtauGenerate.dll` Player API via `pythonnet` (`clr`) to render the USTX to a WAV. Looks in `API/UtauGenerate/bin/{Release,Debug}/net8.0/`. Uses the `ArpasingPlusPhonemizer`. Raises `RuntimeError` with a clear message if pythonnet/the DLL is unavailable, suggesting `--prior-wav`.
- `extract_prior_mel(wav_path)` — Loads WAV at 24 kHz via librosa, calls SoulX-Singer's `mel_to_soulx_mel` to produce a normalized log-mel of shape `(T, 128)`.
- `extract_f0_rmvpe(wav_path, rmvpe_model_path, device)` — Uses SoulX-Singer's `F0Extractor` (RMVPE) at `target_sr=24000`, `hop=480`. Returns `(T,)` F0 in Hz with `0` = unvoiced.
- `synthesize_f0_from_midi(notes_data, total_frames)` — Coarse fallback: paints constant `freq = 440 * 2^((tone-69)/12)` over each note's frame range. Used when no F0 file and no RMVPE checkpoint are available.
- `extract_or_load_f0(...)` — Legacy fallback chain: (1) `--f0` `.npy` file, (2) RMVPE if checkpoint exists, (3) MIDI-pitch fallback. Superseded by the `--f0-mode` strategy dispatch in `main()`.
- `_interpolate_pitch_points(pitch_points, t_ms)` — Linearly interpolates cents deviation from sorted USTX pitch control points at time `t_ms`. Extrapolates flat beyond endpoints.
- `synthesize_f0_from_ustx_pitch(notes_data, total_frames, voicing=None)` — Synthesizes F0 from USTX per-note pitch control points. For each note, computes base Hz from MIDI tone, interpolates pitch points to get cents deviation per frame, converts `base_hz * 2^(cents/1200)` to Hz. Zeros out unvoiced frames if `voicing` mask is provided.
- `get_note_coverage_voicing(notes_data, total_frames)` — Derives voicing from MIDI note coverage. A frame is voiced iff it falls within any note's time range; frames in silence gaps between notes are unvoiced.
- `_extract_conditioning_ustx(notes_data, T, phoneset_path)` — Strategy function for `ustx` mode: voicing from note coverage → F0 from pitch curves → phonemes. Returns `(f0, voicing, phoneme_ids)`.
- `_extract_conditioning_f0_based(f0_raw, notes_data, T, phoneset_path)` — Strategy function for `rmvpe`/`midi`/`file` modes: resamples raw F0 → voicing from `f0 > 0` → builds phonemes. Returns `(f0, voicing, phoneme_ids)`.
- `_F0_EXTRACTORS` — Dispatch dict mapping `"file"`, `"rmvpe"`, `"midi"` to raw F0 extraction callables.
- `_lyric_to_word(lyric)` — Strips USTX bracket hints (`word[hint]`), maps continuation/rest tokens (`+`, `-`, `R`, `r`, empty) to `<SP>`.
- `build_phoneme_ids(notes, ms_per_tick, total_frames, phoneset_path)` — Calls SoulX-Singer's `g2p_transform(words, lang="English")` to convert lyrics to `en_PH1-en_PH2-...` strings, then `_build_mel2note` to expand into per-frame phoneme IDs and a phoneme mask. Resolves phoneme indirection via VocaloFlow's `resolve_phoneme_indirection` and resamples to `total_frames` with `mode="nearest"`. Warns if the phoneme mask is empty.
- `load_model(checkpoint_path, device)` — Loads a checkpoint with `weights_only=False`, instantiates `VocaloFlow(config)` from the embedded `config` (or default `VocaloFlowConfig`), and **prefers `ema_model_state_dict`** over `model_state_dict` if present. Uses `strict=False` to allow loading v2 checkpoints into v3 models (missing ConvNeXt keys are randomly initialized; logs missing/unexpected keys).
- `build_plbert_frame_features(notes, ms_per_tick, total_frames, phoneset_path, plbert_extractor)` — Builds per-frame PL-BERT features from USTX notes at inference time. Converts lyrics to phonemes via g2p, maps to IPA, runs PL-BERT, and expands to frame level. Returns `(total_frames, 768)` numpy array.
- `infer_chunked(model, prior_mel, f0, voicing, phoneme_ids, chunk_size=256, overlap=16, num_steps=32, method="midpoint", device, cfg_scale=1.0, plbert_features=None, speaker_embedding=None, time_schedule="sway", sway_coeff=-1.0, atol=1e-5, rtol=1e-5)` �� Splits the sequence into overlapping windows of size `chunk_size`, pads short tails to `chunk_size` (with a `padding_mask` so the model ignores the padding), runs `sample_ode` per chunk, and crossfades chunks via a linear ramp on the `overlap` frames at each side. For sequences shorter than one chunk, runs a single padded pass. Passes `time_schedule`, `sway_coeff`, `atol`, and `rtol` through to `sample_ode`.
- `_build_chunk_window(ci, start, end, length, total_frames, overlap, chunk_size, starts, tail_chunk_idx)` — Builds the per-chunk blending window. Regular chunks get linear fade-in/fade-out at overlap boundaries. The tail chunk (appended when the sequence doesn't end on a chunk boundary) uses a zero-weight region before the crossfade, then a linear fade-in aligned with the previous chunk's fade-out, then full weight to the end. This avoids the large averaging region that a naive overlap would create.
- `mel_to_wav(mel)` — Transposes `(T, 128) -> (128, T)` and calls SoulX-Singer's `invert_mel_to_audio_soulxsinger` (Vocos) to produce 24 kHz audio.
- `parse_args()` / `main()` — CLI driver. `main()` runs steps 1–10. Steps 4–6 (F0, voicing, phonemes) are dispatched via strategy functions based on `--f0-mode`: `ustx` (default) uses `_extract_conditioning_ustx` (voicing from note coverage → F0 from USTX pitch curves → phonemes); other modes use `_extract_conditioning_f0_based` (F0 first → voicing from F0 → phonemes). Prints `[diag]` lines after every major stage.

**CLI flags:**
| Flag | Default | Notes |
|---|---|---|
| `--ustx` | required | Input `.ustx` |
| `--checkpoint` | required | VocaloFlow `.pt` |
| `--prior-wav` | None | Pre-rendered WAV; if missing, tries the C# render fallback |
| `--f0` | None | Pre-extracted `.npy` F0 in Hz (implies `--f0-mode file`) |
| `--f0-mode` | `ustx` | `ustx` = pitch curves from USTX + phoneme voicing; `rmvpe` = extract from prior WAV; `midi` = flat MIDI pitches; `file` = load from `--f0` .npy |
| `--output` | `output.wav` | |
| `--device` | `auto` | `auto`/`cuda`/`cpu` |
| `--num-ode-steps` | `16` | |
| `--ode-method` | `midpoint` | `euler`, `midpoint`, `rk4`, or `dopri5` |
| `--chunk-size` | `125` | 125 frames ≈ 2.5 s (matches training distribution mean) |
| `--overlap` | `16` | 16 frames ≈ 0.32 s crossfade |
| `--rmvpe-model` | `SoulX-Singer/pretrained_models/SoulX-Singer-Preprocess/rmvpe/rmvpe.pt` | |
| `--phoneset` | `SoulX-Singer/soulxsinger/utils/phoneme/phone_set.json` | |
| `--cfg-scale` | `2.0` | Classifier-free guidance; `1.0` disables CFG |
| `--time-schedule` | `sway` | `uniform` or `sway` timestep distribution (ignored by dopri5) |
| `--sway-coeff` | `-1.0` | Sway coefficient *s* (only used with `--time-schedule sway`) |
| `--atol` | `1e-5` | Absolute tolerance for adaptive methods (dopri5) |
| `--rtol` | `1e-5` | Relative tolerance for adaptive methods (dopri5) |
| `--plbert-dir` | `../PL-BERT` | Path to PL-BERT directory (for live feature extraction when `model.config.use_plbert=True`) |
| `--mask-phonemes` | off | Diagnostic flag: zeros out all phoneme IDs after `build_phoneme_ids`, removing linguistic conditioning. Useful for isolating the model's behavior driven by F0/voicing/prior alone. |
| `--save-mels` | off | Saves `{stem}_prior_mel.npy` and `{stem}_mel.npy` alongside the output WAV (both `(T, 128)` float32 arrays in normalized log-mel space). Used by `DataAnalysis/mel_analysis.ipynb` for visualization. |

### `inference.py` — ODE sampler

**Architecture:** Modular stepper-function design. Each fixed-step method is a standalone function with signature `(x_t, t_val, dt, get_velocity, device, batch_size) -> (x_new, v_last)`. A registry dict `_FIXED_STEP_METHODS` maps method names to functions for clean dispatch. Adaptive Dopri5 has its own integration function since it controls its own step sizes.

**Key symbols:**
- `_build_time_schedule(num_steps, schedule="sway", sway_coeff=-1.0)` — Returns `num_steps+1` time values spanning `[0, 1]`. `"uniform"` = evenly spaced; `"sway"` applies `t + s*(cos(π/2 * t) - 1 + t)` to concentrate steps near endpoints.
- `_step_euler(...)` — 1 velocity eval per step.
- `_step_midpoint(...)` — 2 velocity evals. Half-step probe, then full step at midpoint velocity.
- `_step_rk4(...)` — 4 velocity evals. Classic 4th-order Runge-Kutta (`(k1 + 2*k2 + 2*k3 + k4) / 6`).
- `_integrate_dopri5(x_0, get_velocity, device, batch_size, h_init, atol, rtol, max_steps=1000, diagnostics)` — Adaptive Dormand-Prince 4(5) with FSAL optimization. Uses embedded error estimate to control step size. Integrates from t=0 to t=1 with automatic step selection. Ignores the time schedule (adapts its own steps). Prints diagnostics at 25/50/75% progress plus total accepted/rejected steps.
- `sample_ode(model, x_0, f0, voicing, phoneme_ids, num_steps=32, method="midpoint", padding_mask=None, diagnostics=True, cfg_scale=1.0, plbert_features=None, speaker_embedding=None, time_schedule="sway", sway_coeff=-1.0, atol=1e-5, rtol=1e-5)` — Main entry point. Dispatches to fixed-step loop (euler/midpoint/rk4) or `_integrate_dopri5`. For dopri5, `num_steps` is used as a hint for initial step size (`h_init = 1/num_steps`).

- **CFG (`cfg_scale > 1.0`):** Each velocity query runs both a conditional pass and an unconditional pass with `f0`, `voicing`, `phoneme_ids`, and `plbert_features` zeroed out, then returns `v_uncond + cfg_scale * (v_cond - v_uncond)`. Doubles cost when enabled.
- **Diagnostics (fixed-step):** Prints `|v|` mean/max and `|x_t|` mean at steps `{0, num_steps//4, num_steps//2, num_steps-1}`.
- **Diagnostics (dopri5):** Prints at 25%, 50%, 75% progress through [0,1], plus summary of accepted/rejected steps.

### `evaluate.py` — Quality metrics
Two helpers used during validation:
- `mel_mse(pred, target, padding_mask=None)` — Mean-squared error over the 128 mel bins, optionally masked.
- `mel_mae(pred, target, padding_mask=None)` — Mean-absolute error, same masking semantics.

Both expect `(B, T, 128)` tensors and a `(B, T)` bool mask where `True` = valid frame.

### `__init__.py`
Empty package marker.

## Example commands

All commands are run from the `VocaloFlow/` directory.

**Standard inference (pre-rendered prior WAV):**
```bash
python -m inference.pipeline 
    --ustx "../demo/short_test.ustx" 
    --prior-wav "../demo/short_test_Track1.wav" 
    --checkpoint "checkpoints/checkpoint_10000.pt" 
    --output "../demo/target_short_test.wav"
```

**Late-checkpoint full run:**
```bash
python -m inference.pipeline \
    --ustx prior.ustx \
    --prior-wav prior.wav \
    --checkpoint checkpoints/checkpoint_200000.pt \
    --output output.wav
```

**Force CPU and Euler integration (fast smoke test):**
```bash
python -m inference.pipeline \
    --ustx "../demo/short_test.ustx" \
    --prior-wav "../demo/short_test_Track1.wav" \
    --checkpoint "checkpoints/checkpoint_20000.pt" \
    --output "../demo/smoke.wav" \
    --device cpu --ode-method euler --num-ode-steps 16
```

**Disable classifier-free guidance:**
```bash
python -m inference.pipeline \
    --ustx "../demo/short_test.ustx" \
    --prior-wav "../demo/short_test_Track1.wav" \
    --checkpoint "checkpoints/checkpoint_20000.pt" \
    --output "../demo/no_cfg.wav" \
    --cfg-scale 1.0
```

**Use a pre-extracted F0 curve (skip RMVPE):**
```bash
python -m inference.pipeline \
    --ustx "../demo/short_test.ustx" \
    --prior-wav "../demo/short_test_Track1.wav" \
    --f0 "../demo/short_test_f0.npy" \
    --checkpoint "checkpoints/checkpoint_20000.pt" \
    --output "../demo/with_f0.wav"
```

**Render the prior via the C# UtauGenerate API (no `--prior-wav`):**
Requires `pythonnet` installed and `API/UtauGenerate/bin/{Release,Debug}/net8.0/UtauGenerate.dll` built.
```bash
python -m inference.pipeline \
    --ustx "../demo/short_test.ustx" \
    --checkpoint "checkpoints/checkpoint_20000.pt" \
    --output "../demo/auto_rendered.wav"
```

**Longer crossfade for very long songs:**
```bash
python -m inference.pipeline \
    --ustx "../demo/long_song.ustx" \
    --prior-wav "../demo/long_song_Track1.wav" \
    --checkpoint "checkpoints/checkpoint_200000.pt" \
    --output "../demo/long_song_out.wav" \
    --chunk-size 512 --overlap 32
```

## Diagnostics — what to look for

The pipeline prints `[diag]` lines after every major stage. Common red flags:
- **`prior_mel near-silent frames > 50%`** — prior WAV is mostly silence; check the render.
- **`voiced ratio` very low (<10%)** — F0 extraction failed, MIDI fallback may be triggering.
- **`phonemes zero/pad > 90%`** — g2p produced nothing usable; lyrics are likely empty/non-English/all rests.
- **`|output - prior| mean ≈ 0`** — model produced ~zero velocity throughout; checkpoint may be untrained or the conditioning shape is wrong.
- **`|v| mean ≈ 0` in `[ode]` lines** — same diagnosis from inside the sampler.
- **`audio peak_db < -60 dBFS`** — output is effectively silent; usually downstream of one of the above.
