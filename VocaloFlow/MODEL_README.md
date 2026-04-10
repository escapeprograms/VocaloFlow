# VocaloFlow Model Documentation

This document is a complete, code-free description of the VocaloFlow model: what it does, what it consumes, what it produces, every architectural choice and training nuance, and the failure modes that have been observed and fixed during development. It is intended to be sufficient on its own — without access to the source — for another person or model to reason about the system, predict its behavior, and diagnose problems.

---

## 1. What VocaloFlow Is

VocaloFlow is a **conditional flow matching** model that maps a low-quality singing-voice mel-spectrogram (rendered by Vocaloid / OpenUTAU using a synthetic voicebank) to a high-quality singing-voice mel-spectrogram (resembling a real human singer). It does **not** generate audio directly — it operates entirely in the mel domain, and a separate vocoder (SoulX-Singer's Vocos) converts the predicted mel back to a waveform.

The model learns a **velocity field** that, when integrated as an ordinary differential equation from t=0 (the prior mel) to t=1 (the target mel), produces the high-quality output. This is the Optimal Transport Conditional Flow Matching (OT-CFM) formulation.

**High-level data flow:**
```
USTX project → OpenUTAU render → prior WAV → prior mel ─┐
                                                        ├→ VocaloFlow → enhanced mel → Vocos vocoder → enhanced WAV
F0, voicing, phonemes (extracted from WAV + USTX) ──────┘
```

---

## 2. Audio and Mel-Spectrogram Conventions

These are fixed throughout the project. **All inputs and outputs are in this representation** unless otherwise stated.

| Property        | Value      | Notes                                              |
|-----------------|-----------:|----------------------------------------------------|
| Sample rate     | 24000 Hz   | Matches SoulX-Singer's Vocos vocoder               |
| Hop size        | 480        | 20 ms per frame                                    |
| Mel channels    | 128        | "SoulX-Singer mel space" (normalized log-mel)      |
| Mel format      | float32    | Per-frame, T frames × 128 bins                     |

A 1-second audio clip becomes 50 mel frames. A typical training chunk is 256 frames = ~5.12 seconds.

The mel values are in SoulX-Singer's normalized log-mel space; the `mel_to_soulx_mel()` and `invert_mel_to_audio_soulxsinger()` functions in DataSynthesizer handle the forward and inverse transforms.

---

## 3. Inputs to the Model

The forward pass takes **seven** named arguments. Five of them are per-frame signals shaped `(B, T)` or `(B, T, 128)`. Two are scalar/auxiliary.

| Name           | Shape          | Dtype   | Range / Meaning                                              |
|----------------|----------------|---------|--------------------------------------------------------------|
| `x_t`          | (B, T, 128)    | float32 | The interpolated mel state at flow timestep t. The variable that is being denoised/transformed. |
| `t`            | (B,)           | float32 | Flow timestep ∈ [0, 1]. 0 = prior, 1 = target.               |
| `prior_mel`    | (B, T, 128)    | float32 | The Vocaloid mel — the source of the OT path. Constant per sample, used as conditioning. |
| `f0`           | (B, T)         | float32 | Fundamental frequency in Hz. 0 means unvoiced.               |
| `voicing`      | (B, T)         | float32 | Binary mask: 1 = voiced frame, 0 = unvoiced.                 |
| `phoneme_ids`  | (B, T)         | int64   | Per-frame token IDs into a 2820-entry phoneme vocabulary. 0 = pad. |
| `padding_mask` | (B, T)         | bool    | True = real frame, False = right-padding to fill the batch.  |

**Important nuances:**

- **`x_t` is not the prior**. During training, `x_t` is constructed as `x_t = (1 - (1-σ)·t)·x_0 + t·x_1` where `x_0 = prior_mel` and `x_1 = target_mel`. At inference, `x_t` evolves via ODE integration starting from `x_0`. The model needs **both** `x_t` and the original `prior_mel` because `x_t` is a blend, and the model has to know the clean prior to predict the right velocity.
- **`f0` and `voicing` are technically redundant**: `voicing == (f0 > 0)`. We pass both because the embedding paths are different and the model can learn to use them differently.
- **`phoneme_ids` arrives already resolved**. Raw data on disk has a `phoneme_mask` that is an *index* into a per-chunk `phoneme_ids` array; the dataset resolves this indirection before producing the tensor the model sees.
- **All five per-frame signals are aligned to the same time axis** (the target_mel's time dimension) by the dataset. Resampling is linear for continuous signals (prior mel, F0) and nearest-neighbor for discrete signals (voicing, phoneme IDs).
- **`B` is batch size, `T` is sequence length** (typically 256 frames during training, variable at inference time chunked to 256 with overlap).

---

## 4. Output

A single tensor of shape `(B, T, 128)`, float32 — the predicted **velocity field** at the input state `x_t`. This is *not* a mel-spectrogram. To produce a mel-spectrogram you must integrate the ODE `dx/dt = v_θ(x_t, t, …)` from t=0 to t=1 starting at `x_0 = prior_mel`. After integration, the resulting `x_1` is the enhanced mel that gets vocoded to audio.

---

## 5. Architecture (v2)

VocaloFlow is a **6-layer Diffusion Transformer (DiT)** with rotary position embeddings, AdaLN-Zero conditioning, and per-stream input normalization. Total parameter count: **~29.2 million**.

### 5.1 Key dimensions

| Hyperparameter      | Value | Notes                                           |
|---------------------|------:|-------------------------------------------------|
| `hidden_dim`        |   512 | Width of the transformer                         |
| `num_heads`         |     8 | Attention heads                                  |
| `head_dim`          |    64 | hidden_dim / num_heads                           |
| `ffn_dim`           |  2048 | FFN intermediate dimension (4× expansion)        |
| `num_dit_blocks`    |     6 | Stacked transformer layers                       |
| `phoneme_embed_dim` |    64 | Per-token phoneme embedding width                |
| `f0_embed_dim`      |    64 | Learned F0 embedding width                       |
| `dropout`           |   0.1 | Applied in every DiT block (attn + FFN)          |
| `max_seq_len`       |   256 | Max frames per chunk (~5.12s)                    |
| `phoneme_vocab_size`|  2820 | Full SoulX-Singer phone_set.json vocabulary      |

### 5.2 Forward pipeline (in order)

**Step 1 — Embed phonemes**
`phoneme_ids: (B, T) int64` → look up in a learned `nn.Embedding(2820, 64, padding_idx=0)` → `(B, T, 64)`. Token ID 0 maps to a zero vector (padding).

**Step 2 — Embed F0**
`f0: (B, T)` → MLP `Linear(1, 64) → SiLU → Linear(64, 64)` → `(B, T, 64)`. Continuous Hz values are projected to a 64-dim dense vector. This replaces the v1 design of using raw F0 as a single channel, which underparameterized the pitch signal.

**Step 3 — Per-stream LayerNorm (CRITICAL)**
Each input stream is independently normalized **before** concatenation:

| Stream     | Channels | Normalized? |
|------------|---------:|-------------|
| `x_t`      |      128 | ✓ LayerNorm(128) |
| `prior_mel`|      128 | ✓ LayerNorm(128) |
| `f0_emb`   |       64 | ✓ LayerNorm(64)  |
| `ph_emb`   |       64 | ✓ LayerNorm(64)  |
| `voicing`  |        1 | ✗ passed raw    |

**Why per-stream norms matter**: raw mel values (typically in [-12, 0]), F0 embeddings (zero-mean Gaussian-ish from a learned MLP), and phoneme embeddings have completely different value ranges. Without per-stream normalization, a single linear projection would have to balance them on its own, and one stream (whichever has the largest scale) would dominate the gradient signal early in training. This is one of the main fixes from v1.

**Step 4 — Concatenate**
`(B, T, 128 + 128 + 64 + 64 + 1)` = `(B, T, 385)`.

**Step 5 — Input projection**
`Linear(385, 512)` → `(B, T, 512)`. This is a plain Linear layer (acts as Conv1d with kernel=1).

**Step 6 — Timestep conditioning vector**
`t: (B,)` → sinusoidal embedding (256 dims, max_period=10000) → `Linear(256, 512) → SiLU → Linear(512, 512)` → `(B, 512)`. This vector `c` modulates every transformer block via AdaLN-Zero (see 5.3). The sinusoidal embedding gives the model a continuous, smoothly-varying representation of t ∈ [0, 1].

**Step 7 — 6× DiT block**
Each block consumes `(B, T, 512)` hidden states and the conditioning vector `(B, 512)`, and produces `(B, T, 512)`. See 5.3 for what happens inside.

**Step 8 — Output projection**
`LayerNorm(512) → Linear(512, 128)` → `(B, T, 128)` velocity prediction.

### 5.3 Inside a single DiT block

A DiT block is a standard pre-norm transformer block, modified in two ways: it uses **rotary position embeddings (RoPE)** for self-attention, and it is **modulated by AdaLN-Zero** instead of having its own learnable LayerNorm parameters.

**AdaLN-Zero conditioning**: The conditioning vector `c` (B, 512) is fed through `SiLU → Linear(512, 6·512)`, which produces six modulation tensors per layer: `(γ₁, β₁, α₁, γ₂, β₂, α₂)`, each of shape `(B, 1, 512)`. The Linear is **zero-initialized** (both weight and bias) so that at the start of training, `γ=0 → scale=1`, `β=0 → shift=0`, `α=0 → gate=0`. This means each DiT block starts as an exact identity function — a critical trick for stable training of deep transformers with adaptive normalization.

**Sub-layer 1 — Self-Attention with AdaLN + RoPE**
1. Pre-norm: `h = LayerNorm(x)` (LayerNorm has `elementwise_affine=False` — no learnable gain/bias because the AdaLN params provide that role).
2. Modulate: `h = (1 + γ₁) · h + β₁`.
3. Compute Q, K, V via a single `Linear(512, 1536)` projection, reshape to `(B, 8 heads, T, 64)`.
4. **Apply RoPE to Q and K** using precomputed `freqs_cis` of shape `(max_len, head_dim/2, 2)`. RoPE rotates pairs of channels in each head by an angle that depends on the position index — this gives the model relative positional information without additive position embeddings, and generalizes to sequence lengths it hasn't seen during training.
5. Compute attention via `F.scaled_dot_product_attention` (uses FlashAttention when available on the GPU).
6. Build attention mask from `padding_mask`: a `(B, 1, T, T)` bool that is True only where both query and key positions are valid. Padding positions are excluded from both directions of attention.
7. Project back: `Linear(512, 512)` → apply **dropout (p=0.1)** → gated residual: `x = x + α₁ · attn_out`.

**Sub-layer 2 — FFN with AdaLN**
1. Pre-norm: `h = LayerNorm(x)`.
2. Modulate: `h = (1 + γ₂) · h + β₂`.
3. FFN: `Linear(512, 2048) → GELU → Linear(2048, 512)`.
4. Apply **dropout (p=0.1)** → gated residual: `x = x + α₂ · ffn_out`.

The dropout in steps 7 and (FFN-)4 is the **only** stochastic regularization in the model architecture (CFG dropout, in section 7, lives in the loss function, not the model).

### 5.4 Why narrow-and-deep instead of wide-and-shallow

The v1 architecture was 2 layers, hidden_dim 1024, FFN 4096 (~35M params). This was found to overfit quickly, because the massive FFN layers act as lookup tables — they have enough capacity to memorize training examples without learning generalizable representations. v2 redistributes the same parameter budget across more sequential processing steps (6 layers, 512/2048), which acts as an implicit regularizer: the model has to do meaningful computation at every layer rather than caching answers.

---

## 6. Training Objective (OT-CFM)

The training loss is **Optimal Transport Conditional Flow Matching**.

### 6.1 The interpolation path

For a paired sample `(x_0 = prior_mel, x_1 = target_mel)`, sample a flow time `t ~ Uniform(0, 1)` and construct the interpolated state:

```
x_t = (1 - (1 - σ) · t) · x_0 + t · x_1
```

where σ = `sigma_min` = 1e-4 (a small constant for numerical stability — at t=0, `x_t = x_0`; at t=1, `x_t ≈ x_1`).

### 6.2 The target velocity

The optimal transport velocity field along this linear interpolation path is **constant in t** for any given (x_0, x_1) pair:

```
v_target = x_1 - (1 - σ) · x_0
```

This is the key property of OT-CFM: there is a single right answer for the velocity at every (x_t, t), so the model has a stable target to regress toward. Compare to score-matching diffusion, where the target is noise-scaled and changes with t.

### 6.3 The loss

```
L = E_{(x_0, x_1, cond), t} [ ‖ v_θ(x_t, t, cond) - v_target ‖² ]
```

Computed as masked mean squared error: differences are summed only over valid (non-padded) frames, then divided by `(num_valid_frames × mel_channels)`.

### 6.4 Inference: ODE integration

At inference, the model has no `x_1` to interpolate against. Instead, it integrates the learned ODE forward from t=0 to t=1:

```
x_t starts at x_0 (the prior mel)
For each step:    v = v_θ(x_t, t, cond)
                  x_t ← x_t + dt · v        (Euler)
                  or use midpoint rule (default, more accurate)
```

Default: 32 ODE steps with the **midpoint method** (each step does two model forward passes — one to estimate the velocity at the midpoint, one to apply it). For long sequences, inference is chunked into 256-frame windows with linear-crossfade overlap (default 16 frames overlap).

---

## 7. Classifier-Free Guidance (CFG)

CFG is implemented as **conditioning dropout during training** and **a guided velocity at inference**.

### 7.1 Training-time CFG dropout

During training, with probability `cfg_dropout_prob` = 0.2, the conditioning signals for an entire sample are zeroed out:

- `f0` → all zeros
- `voicing` → all zeros
- `phoneme_ids` → all zeros (all PAD)
- `prior_mel` is **NOT** dropped (it's the core signal)
- `x_t` is **NOT** dropped (it's the noisy state, not conditioning)

This trains the model to produce a sensible velocity even without conditioning, which is needed for the unconditional path at inference.

### 7.2 Inference-time guidance

At inference, with `cfg_scale > 1.0`, every ODE step does **two** forward passes:

```
v_cond   = model(x_t, t, x_0, f0,    voicing,    phoneme_ids,    mask)
v_uncond = model(x_t, t, x_0, zeros, zeros,      zeros,          mask)
v_guided = v_uncond + cfg_scale · (v_cond - v_uncond)
```

`cfg_scale = 1.0` disables guidance entirely (single forward pass per step). Default is 2.0 — higher values amplify the conditioning's influence at the cost of doubled inference compute and a risk of over-saturation artifacts.

### 7.3 Subtle bug class: CFG dropout in eval

`FlowMatchingLoss` is implemented as an `nn.Module`, so it has its own `.training` flag separate from the model's. Calling `model.eval()` does **not** put the loss into eval mode. If validation forgets to call `criterion.eval()`, CFG dropout fires for ~20% of validation samples and inflates val loss artificially. The fix is to wrap the validation loop in a `try/finally` that toggles `criterion.eval()` before and `criterion.train()` after.

---

## 8. Regularization Techniques

| Technique           | Where             | Strength | Purpose                                          |
|---------------------|-------------------|---------:|--------------------------------------------------|
| Dropout             | Every DiT block (attn out, FFN out) | p=0.1 | Generic transformer regularization        |
| CFG dropout         | Loss function     | p=0.2  | Trains unconditional path, prevents over-reliance on any single conditioning signal |
| Weight decay        | AdamW optimizer   | 0.01   | L2 regularization                                |
| Gradient clipping   | After backward    | norm=1 | Prevents loss spikes during early training        |
| Per-stream LayerNorm| Input stage       | always | Prevents one input stream from dominating         |
| EMA model           | Outside optimizer step | decay=0.9999 (with warmup, see 9.2) | Smoothed weights for inference, lower variance val loss |

---

## 9. Optimization and EMA

### 9.1 Optimizer

- AdamW
- Peak learning rate: 1e-4
- Weight decay: 0.01
- Linear warmup over 1000 steps, then cosine decay to 0 over the remaining steps
- Gradient norm clipping at 1.0
- Total steps: 200,000 (configurable)

### 9.2 EMA model with warmup

In addition to the model being optimized, a second copy ("EMA model") is maintained. After every optimizer step, the EMA weights are updated as:

```
effective_decay = min(0.9999, (step + 1) / (step + 10))
p_ema ← lerp(p_ema, p_model, 1 - effective_decay)
```

The **warmup factor `(step+1)/(step+10)`** is critical. Without it, the EMA decay would be a flat 0.9999 from step 0, which means the EMA model is `0.9999^N` random initialization at step N. At step 1000 the EMA is still 90% random init; at step 10000 it's 37% random init. This was actually observed in development as a bug: validation loss sat near 2.0 (the random-init baseline) for thousands of steps while training loss dropped to ~0.4. With warmup:

| Step | effective_decay | EMA composition                                     |
|------|----------------:|-----------------------------------------------------|
| 0    | 0.10            | tracks live model very tightly                      |
| 100  | 0.918           | still tight                                         |
| 1000 | 0.991           | smoothing kicks in                                  |
| 10k  | 0.9991          | nearly converged to target                          |
| 100k | 0.9999 (clamped)| full target decay                                   |

**The EMA model is what gets used for inference.** Checkpoints contain both the live and EMA state dicts; the loader prefers EMA if present.

### 9.3 Validation

Validation runs the EMA model under `torch.no_grad()` over the entire val set with `model.eval()` (disables dropout) and `criterion.eval()` (disables CFG dropout). Returns the average per-batch loss. Validation runs every `val_every` steps (default 100, configurable).

---

## 10. Data Pipeline

### 10.1 Per-chunk files on disk

Each training chunk has six `.npy` files plus a row in `manifest.csv`:

| File              | Shape       | Dtype | Notes                                                 |
|-------------------|-------------|-------|-------------------------------------------------------|
| `target_mel.npy`  | (T_target, 128) | float32 | High-quality singer mel                          |
| `prior_mel.npy`   | (T_prior, 128)  | float32 | Vocaloid mel (different T from target!)          |
| `target_f0.npy`   | (T_target,)     | float32 | F0 in Hz, 0 = unvoiced                           |
| `target_voicing.npy` | (T_target,)  | float32 | Binary V/UV mask                                 |
| `phoneme_ids.npy` | (P,)            | int32 | Per-chunk phoneme token sequence                  |
| `phoneme_mask.npy`| (T_mask,)       | int32 | Per-frame INDEX into phoneme_ids (indirection!)   |

**Two critical pitfalls** the dataset class handles automatically:
1. **The phoneme indirection**: `phoneme_mask[t]` is *not* a phoneme token ID, it's an *index* into `phoneme_ids`. The actual token is `phoneme_ids[phoneme_mask[t]]`. Forgetting this gives garbage embeddings.
2. **Time dimension mismatches**: `prior_mel`, `target_mel`, and `phoneme_mask` may all have different time dimensions. The dataset resamples everything to `target_mel`'s T (linear interp for continuous signals, nearest-neighbor for discrete).

### 10.2 In-memory transformation per sample

1. Load all six `.npy` files.
2. Resolve phoneme indirection.
3. Resample everything to `target_mel`'s T.
4. Crop or pad to `max_seq_len = 256`:
   - If T > 256 and training: random crop start.
   - If T > 256 and eval: start crop (deterministic).
   - If T < 256: zero-pad on the right.
5. Build a `padding_mask` (True for the `length = min(T, 256)` real frames, False for padding).
6. Return a dict with all tensors at length 256.

Manifest filtering: rows with `dtw_cost > max_dtw_cost` (default 100) are excluded — these are chunks where the alignment between Vocaloid and human voice was too poor for the pair to be useful training signal.

### 10.3 Train/val split

Two strategies, switchable via `config.split_mode`:

- **`"song"` (default)**: split by `dali_id`. All chunks from a given song go entirely to either train or val. This is the strict, leakage-free split — val songs are completely held out.
- **`"random"`**: split by individual chunk. Chunks from the same song can appear on both sides. Useful as a baseline to compare against, since it eliminates "song difficulty" as a confounder.

Default `val_fraction = 0.2` (20% of songs).

---

## 11. Logging and Checkpointing

- **Checkpoints** contain `step`, `model_state_dict` (live), `ema_model_state_dict`, `optimizer_state_dict`, and the full `config` dataclass. Saved every `save_every` (default 5000) steps. Inference loads the EMA dict if present.
- **TensorBoard**: scalars `train/loss`, `train/lr`, `val/loss` written to `config.log_dir`. View with `tensorboard --logdir logs`.
- **Weights & Biases**: same scalars + gradient histograms via `wandb.watch`. Project = `archimedesli/VocaloFlow`.

---

## 12. Diagnostic Playbook

Every issue observed during development falls into one of these patterns. If validation or training behavior looks wrong, work through the table.

### 12.1 Validation loss is much higher than training loss

| Symptom                                      | Likely cause                                                  | How to test                                          |
|----------------------------------------------|---------------------------------------------------------------|------------------------------------------------------|
| Val loss stuck near random-init baseline (~2.0) for thousands of steps while train loss drops | **EMA warmup missing** — EMA is still mostly random init | Compute `(step+1)/(step+10)` and check it's being used; or run validate() on the live model and compare |
| Val loss is ~10–30% above train loss, drops in lockstep | **CFG dropout firing during validation** | Check `criterion.eval()` is called in `validate()` |
| Val loss is ~5% above train loss, both decreasing smoothly | **Real, healthy generalization gap** (dropout off in eval gives the model full capacity, but val data is genuinely held out) | This is fine. No fix needed. |
| Val loss > train loss only on `split_mode="song"` | **Real generalization gap from unseen songs** | Compare against `split_mode="random"`; if random matches train, the model is overfitting to specific singers/songs |

**Diagnostic recipe** (already used and removed from `train.py` once confirmed): temporarily add a third validation pass that runs `validate()` on the **live** model instead of the EMA model. If `val/loss_live ≈ train/loss` but `val/loss (EMA) >> train/loss`, the gap is purely EMA lag. If `val/loss_live` is *also* high, look for a real bug in data, loss, or model.

### 12.2 Training loss does not decrease at all

| Symptom                                | Likely cause                                                    |
|----------------------------------------|-----------------------------------------------------------------|
| Loss oscillates around the random-init value | LR too high, exploding gradients, or loss target wrong sign |
| Loss decreases for ~10 steps then NaN  | LR too high, missing gradient clipping, or numerical underflow in `(1 - σ) · t` (check σ is positive) |
| Loss is exactly constant               | `requires_grad` lost somewhere, optimizer stepping wrong params, or model in eval mode |

### 12.3 Inference produces silence or noise

| Symptom                                | Likely cause                                                    |
|----------------------------------------|-----------------------------------------------------------------|
| Output mel ≈ prior mel                 | Model is predicting velocity ≈ 0; check `cfg_scale ≥ 1.0`, check the model isn't loading EMA before EMA had warmed up enough, check for sign error in v_target |
| Output mel ≈ random noise              | Wrong checkpoint loaded, model architecture mismatch with checkpoint, or input mel not in normalized SoulX-Singer space |
| Output sounds like prior, not enhanced | Insufficient training, `cfg_scale=1.0` (no guidance), or `num_ode_steps` too small for the midpoint method |
| Output sounds buzzy or artifacted      | Vocoder mismatch — check sample rate and hop size match across mel extraction, model, and vocoder (must be 24000 / 480) |

### 12.4 Bugs already found and fixed (do not re-introduce)

1. **CFG dropout active during validation** — `FlowMatchingLoss.training` is independent of `model.training`. Fixed by `criterion.eval()` + `try/finally` in `validate()`.
2. **EMA stays at random init** — flat `decay=0.9999` from step 0 means `0.9999^N ≈ random`. Fixed with warmup `min(0.9999, (step+1)/(step+10))`.
3. **Phoneme dimension dominated input (v1)** — phoneme embeddings were 256-dim while prior mel was only 128, so the model learned to over-rely on phonemes. Fixed by reducing phoneme embed to 64 and applying per-stream LayerNorm.
4. **Wide-and-shallow architecture overfits (v1)** — 2 layers × 1024 hidden × 4096 FFN memorized too easily. Fixed by going to 6 × 512 × 2048 (similar param count, much more sequential depth).
5. **F0 underparameterized (v1)** — single channel of raw Hz values. Fixed by learned 64-dim F0 embedding.
6. **`max_dtw_cost=200` admitted bad alignments (v1)** — Lowered to 100 in v2 to filter more aggressively for high-quality pairs.

---

## 13. Quick Reference: Tensor Shape Cheat Sheet

For a batch size B and sequence length T = `max_seq_len = 256`:

| Stage                          | Shape                  | Notes                              |
|--------------------------------|------------------------|------------------------------------|
| `phoneme_ids` (input)          | (B, 256)               | int64                              |
| `phoneme_embed` output         | (B, 256, 64)           | after embedding lookup             |
| `f0` (input)                   | (B, 256)               | float32                            |
| `f0_embed` output              | (B, 256, 64)           | after MLP                          |
| `x_t` (input)                  | (B, 256, 128)          | float32                            |
| `prior_mel` (input)            | (B, 256, 128)          | float32                            |
| `voicing` (input)              | (B, 256)               | float32 (binary)                   |
| Concatenated input             | (B, 256, 385)          | 128+128+64+64+1                    |
| After `input_proj`             | (B, 256, 512)          | hidden representation              |
| `t` (input)                    | (B,)                   | float32 ∈ [0, 1]                   |
| Sinusoidal timestep embedding  | (B, 256)               | (256 = sinusoidal_dim)             |
| Conditioning vector `c`        | (B, 512)               | after timestep_mlp                 |
| Inside DiT block: Q, K, V      | (B, 8, 256, 64)        | (heads, T, head_dim)               |
| Inside DiT block: AdaLN params | 6 × (B, 1, 512)        | γ₁,β₁,α₁,γ₂,β₂,α₂                  |
| After all 6 DiT blocks         | (B, 256, 512)          | hidden                             |
| Output velocity                | (B, 256, 128)          | float32                            |

---

## 14. Glossary

- **OT-CFM** — Optimal Transport Conditional Flow Matching. The training objective. The optimal transport path between two distributions for the L2 cost is a straight line, and the velocity along that line is constant — which gives a stable, low-variance regression target.
- **DiT** — Diffusion Transformer. A transformer architecture (like the original ViT/GPT) augmented with timestep conditioning via AdaLN. Originally introduced for image diffusion by Peebles & Xie 2022.
- **AdaLN-Zero** — Adaptive Layer Normalization with zero-initialized projections. Conditioning is injected by predicting per-layer scale (γ), shift (β), and gate (α) from a conditioning vector. The predictor is zero-initialized so each block starts as identity.
- **RoPE** — Rotary Position Embeddings. Position information is injected by rotating Q and K vector pairs by a position-dependent angle, instead of adding an absolute position embedding. Generalizes to unseen sequence lengths.
- **EMA** — Exponential Moving Average of model weights. A smoothed version of the live model used for inference and validation; produces more stable, less noisy outputs than the live optimizer state.
- **CFG** — Classifier-Free Guidance. Trains both a conditional and unconditional version of the model (via random conditioning dropout), then at inference interpolates between their predictions with a scale factor > 1 to amplify conditioning influence.
- **Velocity field** — The function `v_θ(x_t, t, cond)` the model predicts. Integrating it from t=0 to t=1 gives the final mel.
- **Phoneme indirection** — On disk, `phoneme_mask` stores per-frame indices into a per-chunk `phoneme_ids` array. The actual token is `phoneme_ids[phoneme_mask[t]]`. The dataset resolves this before producing the model input.
- **Prior mel** — The Vocaloid-rendered mel-spectrogram (`x_0`). The starting point of the OT path.
- **Target mel** — The high-quality singer mel-spectrogram (`x_1`). The endpoint of the OT path. Only available during training.
