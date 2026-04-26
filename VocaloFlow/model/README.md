# VocaloFlow Model Memory Palace

Neural network architecture for the VocaloFlow conditional flow matching model (v4 — optional ConvNeXt and/or WaveNet pre-processing, with blurred phoneme boundaries).

## vocaloflow.py

### `VocaloFlow(config: VocaloFlowConfig)`
Top-level nn.Module. Orchestrates all sub-components. Baseline (no pre-processing) ~26M params. Adding ConvNeXt adds ~8.4M. Adding WaveNet (8 blocks, skip=512) adds ~21.7M.

**Forward signature**: `forward(x_t, t, prior_mel, f0, voicing, phoneme_ids, padding_mask=None, plbert_features=None, speaker_embedding=None) -> Tensor`
- Inputs: x_t (B,T,128), t (B,), prior_mel (B,T,128), f0 (B,T), voicing (B,T), phoneme_ids (B,T) int64, padding_mask (B,T) bool, plbert_features (B,T,768) float optional, speaker_embedding (B,192) float optional
- Output: (B,T,128) predicted velocity vector

**Constructor**: Selects `BlurredPhonemeEmbedding` or `PhonemeEmbedding` based on `config.phoneme_blur_enabled`. Creates `ConvNeXtStack` if `config.num_convnext_blocks > 0` else `None`. Creates `WaveNetStack` if `config.num_wavenet_blocks > 0` else `None`. Both pre-processors can independently toggle on or off; if both are on, ConvNeXt runs first. If `config.use_plbert`, creates `plbert_proj = Linear(768, 64)`. If `config.use_speaker_embedding`, creates `speaker_proj = Linear(192, 512)` with zero-init. If `config.voicing_embed_dim > 1`, creates `VoicingEmbedding(dim)` + `LayerNorm(dim)`.

**Input dimension**: Computed dynamically as `mel_channels*2 + f0_embed_dim + phoneme_embed_dim + voicing_embed_dim`. Default with voicing_embed_dim=1: 385. With voicing_embed_dim=32: 416.

**Forward flow**:
1. Phoneme conditioning: if `use_plbert` and `plbert_features` is provided, `plbert_proj(plbert_features)` → (B,T,64); otherwise `phoneme_embed(phoneme_ids)` → (B,T,64).
2. `f0_embed(f0)` → (B,T,64) via learned MLP
3. Per-stream LayerNorm on x_t (128), prior_mel (128), f0_emb (64), ph_emb (64).
4. Voicing: if `voicing_embed_dim > 1`, `norm_voicing(voicing_embed(voicing))` → (B,T,voicing_embed_dim); else raw `voicing.unsqueeze(-1)` → (B,T,1).
5. Concatenate all → (B,T, input_dim)
6. `input_proj` Linear(input_dim, 512) → (B,T,512)
7. `timestep_mlp(t)` → (B,512) conditioning vector `c`. If `use_speaker_embedding`, `c += speaker_proj(speaker_embedding)`.
8a. `convnext_stack(h)` → optional local temporal features (skipped if None)
8b. `h = h + wavenet_stack(h, c)` — optional timestep-conditioned WaveNet residuals (skipped if None)
9. Nx `DiTBlock(h, c, mask)` → (B,T,512) (N = `config.num_dit_blocks`)
10. `output_norm` + `output_proj` Linear(512,128) → (B,T,128)

## dit_block.py

### `AdaLNZero(hidden_dim)`
Adaptive Layer Normalization with zero initialization. Projects conditioning vector `c` (B, hidden_dim) through SiLU + Linear to produce 6 modulation parameters: gamma1, beta1, alpha1 (for attention), gamma2, beta2, alpha2 (for FFN). Each is (B, 1, hidden_dim).

**Key**: Linear layer is zero-initialized so gamma=0→scale=1, beta=0→shift=0, alpha=0→gate=0. This makes each DiT block start as an identity function, critical for stable training.

### `DiTBlock(hidden_dim=512, num_heads=8, ffn_dim=2048, max_len=512, dropout=0.1)`
Pre-norm transformer block with:
- **Self-attention**: 8 heads, 64 dim/head. Q,K,V via single `qkv` Linear. RoPE applied to Q and K. Uses `F.scaled_dot_product_attention` (FlashAttention when available). Dropout on attention output. Gated residual: `x = x + alpha1 * attn_out`.
- **FFN**: Linear(512,2048) → GELU → Linear(2048,512). Dropout on FFN output. Gated residual: `x = x + alpha2 * ffn_out`.
- **LayerNorms**: `elementwise_affine=False` — modulation comes from AdaLN instead.
- **Dropout**: `attn_dropout` and `ffn_dropout` applied after respective output projections, before gated residual.
- **Padding mask**: Converted to (B,1,T,T) attention mask for SDPA.
- **RoPE frequencies**: Precomputed and stored as a buffer (`freqs_cis`).

## embeddings.py

### `sinusoidal_timestep_embedding(t, dim, max_period=10000.0) -> Tensor`
Standard sinusoidal embedding for continuous timestep t. Returns (B, dim) with cos/sin pairs.

### `TimestepMLP(hidden_dim=512, sinusoidal_dim=256)`
Sinusoidal(256) → Linear(256,512) → SiLU → Linear(512,512). Maps scalar timestep t ∈ [0,1] to conditioning vector (B, 512).

### `F0Embedding(embed_dim=64)`
Learned MLP for continuous F0 values: Linear(1,64) → SiLU → Linear(64,64). Maps (B,T) Hz values to (B,T,64) dense embeddings. Provides a richer pitch representation than a single raw channel.

### `VoicingEmbedding(embed_dim=32)`
Learned embedding for binary voiced/unvoiced flag. `nn.Embedding(2, embed_dim)` lookup table. Casts float voicing input to long for the lookup. When `config.voicing_embed_dim > 1`, this replaces the raw 1-dim scalar voicing input. Gets its own `LayerNorm` in the model, following the per-stream normalization pattern. When `voicing_embed_dim == 1` (default), this class is not instantiated and voicing is passed as `voicing.unsqueeze(-1)`.

### `PhonemeEmbedding(vocab_size=2820, embed_dim=64)`
Base class. `nn.Embedding` lookup table with `padding_idx=0` (PAD token). Maps (B,T) int64 → (B,T,64). Reduced from 256 to 64 so phoneme identity doesn't dominate the input over the prior mel signal.

### `BlurredPhonemeEmbedding(PhonemeEmbedding)` — extends PhonemeEmbedding
Adds duration-proportional boundary blending. Near phoneme boundaries, produces a weighted average of adjacent phoneme embeddings instead of hard lookup. Same interface: (B,T) int64 → (B,T,64).

**Algorithm**: Detects boundaries where `ids[:, t] != ids[:, t+1]`, computes per-segment durations, then creates linear blend regions of radius `blend_fraction * min(left_dur, right_dur)` around each boundary. Max blend weight is 0.5 at the boundary itself, tapering to 0 at the edges of the blend window.

**Key properties**:
- `blend_fraction` (float, default 0.3): controls blend region size relative to shorter phoneme duration
- Reuses `self.embedding` from parent class (no weight duplication)
- All-same IDs (e.g., CFG dropout zeros) produce no blurring (correct for unconditional path)
- Short phonemes get proportionally smaller blend windows
- Toggled via `config.phoneme_blur_enabled`

## convnext.py

### `GlobalResponseNorm(dim)`
ConvNeXtV2's inter-channel normalization. Computes per-channel L2 norm across time, normalizes by mean norm, applies learnable scale (`gamma`) and shift (`beta`), plus residual. Initialized to identity (gamma=0, beta=0).

### `ConvNeXtV2Block(dim=512, kernel_size=7, expansion=4, dropout=0.1)`
Single ConvNeXtV2 block: depthwise Conv1d (channels-first) → LayerNorm → pointwise expand (4x) → GELU → GRN → pointwise project → dropout → residual. ~2.1M params per block.

Kernel size 7 at 20ms/frame covers ~140ms — the duration of a consonant cluster or fast syllable transition.

### `ConvNeXtStack(dim=512, num_blocks=4, kernel_size=7, expansion=4, dropout=0.1)`
Sequential stack of ConvNeXtV2Blocks. With 4 blocks, effective receptive field is ~25 frames (~500ms). Does NOT receive timestep conditioning — processes input features independently of flow position. ~8.4M params total.

## wavenet.py

Optional pre-processing stack alternative to ConvNeXt. Follows the non-causal WaveNet denoiser of Parallel WaveGAN / DiffSinger: dilated convolutions with gated (tanh * sigmoid) activations, per-layer timestep conditioning injected before the gate, and accumulated skip connections. Channels-first internally; transposes only at the outer boundary.

### `WaveNetResidualBlock(channels, cond_channels, skip_channels, kernel_size, dilation, dropout=0.1)`
Single residual block. Pipeline: input dropout → dilated Conv1d(C→2C, same-padding) → add per-block conditioning Conv1x1(C_cond→2C) → split into (xa, xb) → `tanh(xa) * sigmoid(xb)` → 1x1 skip and 1x1 out projections → residual add scaled by `sqrt(0.5)`. Returns `(x_out, skip)`. Conditioning projection is per-block (not shared) so different layers can specialise in different timestep regimes.

### `WaveNetStack(hidden_channels=512, cond_channels=512, skip_channels=512, kernel_size=3, n_layers=8, dilation_cycle=4, dropout=0.1)`
Input 1x1 → N residual blocks with cyclic dilations (`2 ** (i % dilation_cycle)`) → ReLU(skip_sum) → Conv1x1 → ReLU → Conv1x1 → output. Takes `(x: (B,T,C), cond: (B,C_cond))`, broadcasts cond across T, and returns `(B,T,C)`. Callers wrap with an outer residual `h = h + wavenet_stack(h, c)` so the DiT blocks see the raw projected input at init time. All Conv1d weights initialised Kaiming-normal (relu). Default 8 blocks at skip=512 ≈ 21.7M params.

**Toggle**: `config.num_wavenet_blocks = 0` disables entirely (stack is `None`). Does not conflict with `num_convnext_blocks`.

## rope.py

### `precompute_freqs_cis(dim, max_len, theta=10000.0) -> Tensor`
Returns (max_len, dim//2, 2) tensor with precomputed [cos, sin] for each position and frequency.

### `apply_rotary_emb(q, k, freqs_cis) -> (q_rot, k_rot)`
Applies rotation to Q and K tensors (B, H, T, D). Splits into even/odd pairs, rotates using the precomputed frequencies, then interleaves back.
