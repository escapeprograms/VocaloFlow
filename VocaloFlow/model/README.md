# VocaloFlow Model Memory Palace

Neural network architecture for the VocaloFlow conditional flow matching model. Three-level class hierarchy:

- **`VocaloFlow`** (base class, `vocaloflow.py`): Shared conditioning encoder (phoneme, F0, voicing, timestep, speaker) and template-method `forward()`.
- **`VocaloFlowHybrid`** (`vocaloflow_hybrid.py`): DiT transformer backbone + optional ConvNeXt/WaveNet pre-processing. Selected via `config.architecture = "hybrid"` (default).
- **`VocaloFlowWaveNet`** (`vocaloflow_wavenet.py`): WaveNet denoiser backbone + optional DiT refinement blocks. Selected via `config.architecture = "wavenet_pure"`. When `wavenet_pure_num_dit_blocks=0`, pure WaveNet with no self-attention.

## vocaloflow.py

### `VocaloFlow(config: VocaloFlowConfig, hidden_dim: int)`
Abstract base class (nn.Module). Owns all shared conditioning modules and provides a template-method `forward()` that calls `_encode_conditioning()` followed by `_backbone_forward()`. Subclasses only implement `_backbone_forward()`.

**Constructor**: Takes `config` and `hidden_dim` (512 for hybrid, 256 for wavenet). Creates all conditioning modules as direct attributes (flat state_dict keys for checkpoint compatibility):
- `phoneme_embed` — `BlurredPhonemeEmbedding` or `PhonemeEmbedding` based on `config.phoneme_blur_enabled`
- `plbert_proj` — `Linear(768, 64)` if `config.use_plbert`
- `f0_embed` — `F0Embedding(64)`
- `voicing_embed` + `norm_voicing` — if `config.voicing_embed_dim > 1`
- `norm_xt`, `norm_prior`, `norm_f0`, `norm_ph` — per-stream LayerNorms
- `input_proj` — `Linear(input_dim, hidden_dim)`
- `timestep_mlp` — `TimestepMLP(hidden_dim=hidden_dim)`
- `speaker_proj` — `Linear(192, hidden_dim)` with zero-init if `config.use_speaker_embedding`

**`_encode_conditioning(...) -> (h, c)`**: Encodes all conditioning inputs. Steps:
1. Phoneme embedding (PL-BERT projection or lookup)
2. F0 embedding via learned MLP
3. Per-stream LayerNorm on x_t, prior_mel, f0_emb, ph_emb
4. Voicing: learned embedding or raw scalar
5. Concatenate all inputs
6. Input projection → h (B,T,hidden_dim)
7. Timestep MLP → c (B,hidden_dim), plus optional speaker embedding additive

**`_backbone_forward(h, c, padding_mask) -> Tensor`**: Abstract — raises NotImplementedError. Subclasses override.

**`forward(x_t, t, prior_mel, f0, voicing, phoneme_ids, padding_mask=None, plbert_features=None, speaker_embedding=None) -> Tensor`**: Template method calling `_encode_conditioning` then `_backbone_forward`. Returns (B,T,128) velocity.

## vocaloflow_hybrid.py

### `VocaloFlowHybrid(config: VocaloFlowConfig)`
Extends `VocaloFlow` with `hidden_dim=config.hidden_dim` (512). Baseline ~26M params. Adding ConvNeXt adds ~8.4M. Adding WaveNet (8 blocks, skip=512) adds ~21.7M.

**Constructor**: Calls `super().__init__(config, hidden_dim=config.hidden_dim)`. Creates hybrid-specific modules:
- `convnext_stack` — `ConvNeXtStack` if `config.num_convnext_blocks > 0`, else None
- `wavenet_stack` — `WaveNetStack` if `config.num_wavenet_blocks > 0`, else None
- `dit_blocks` — `nn.ModuleList` of N `DiTBlock`s (N = `config.num_dit_blocks`)
- `output_norm` — `LayerNorm(hidden_dim)`
- `output_proj` — `Linear(hidden_dim, mel_channels)`

**`_backbone_forward(h, c, padding_mask)`**:
1. Optional ConvNeXt pre-processing (no timestep conditioning)
2. Optional WaveNet pre-processing with outer residual: `h = h + wavenet_stack(h, c)`
3. DiT blocks: `for block in dit_blocks: h = block(h, c, padding_mask)`
4. Output: `output_proj(output_norm(h))` → (B,T,128)

## vocaloflow_wavenet.py

### `VocaloFlowWaveNet(config: VocaloFlowConfig)`
Extends `VocaloFlow` with `hidden_dim=config.wavenet_pure_residual_channels` (256). ~14M params (256ch, no DiT), ~22M (384ch, no DiT).

**Constructor**: Calls `super().__init__(config, hidden_dim=rc)`. Creates wavenet-specific modules:
- `denoiser` — `WaveNetDenoiser` with `output_channels=rc` if DiT follows, else `mel_channels`
- `dit_blocks` — optional `nn.ModuleList` of DiT refinement blocks (if `wavenet_pure_num_dit_blocks > 0`)
- `output_norm` + `output_proj` — only if DiT blocks present

**`_backbone_forward(h, c, padding_mask)`**:
1. WaveNet backbone: `h = denoiser(h, c, padding_mask)`
2. Optional DiT refinement: `for block in dit_blocks: h = block(h, c, padding_mask)`
3. If DiT present: `output_proj(output_norm(h))` → (B,T,128). Else: return h directly.

**No outer residual on WaveNet**: Unlike hybrid's `h = h + wavenet_stack(h, c)`, here WaveNet IS the primary backbone.

## dit_block.py

### `AdaLNZero(hidden_dim)`
Adaptive Layer Normalization with zero initialization. Projects conditioning vector `c` (B, hidden_dim) through SiLU + Linear to produce 6 modulation parameters: gamma1, beta1, alpha1 (for attention), gamma2, beta2, alpha2 (for FFN). Each is (B, 1, hidden_dim).

**Key**: Linear layer is zero-initialized so gamma=0->scale=1, beta=0->shift=0, alpha=0->gate=0. This makes each DiT block start as an identity function, critical for stable training.

### `DiTBlock(hidden_dim=512, num_heads=8, ffn_dim=2048, max_len=512, dropout=0.1)`
Pre-norm transformer block with:
- **Self-attention**: 8 heads, 64 dim/head. Q,K,V via single `qkv` Linear. RoPE applied to Q and K. Uses `F.scaled_dot_product_attention` (FlashAttention when available). Dropout on attention output. Gated residual: `x = x + alpha1 * attn_out`.
- **FFN**: Linear(512,2048) -> GELU -> Linear(2048,512). Dropout on FFN output. Gated residual: `x = x + alpha2 * ffn_out`.
- **LayerNorms**: `elementwise_affine=False` — modulation comes from AdaLN instead.
- **Padding mask**: Converted to (B,1,T,T) attention mask for SDPA.
- **RoPE frequencies**: Precomputed and stored as a buffer (`freqs_cis`).

## embeddings.py

### `sinusoidal_timestep_embedding(t, dim, max_period=10000.0) -> Tensor`
Standard sinusoidal embedding for continuous timestep t. Returns (B, dim) with cos/sin pairs.

### `TimestepMLP(hidden_dim=512, sinusoidal_dim=256)`
Sinusoidal(256) -> Linear(256,512) -> SiLU -> Linear(512,512). Maps scalar timestep t in [0,1] to conditioning vector (B, 512).

### `F0Embedding(embed_dim=64)`
Learned MLP for continuous F0 values: Linear(1,64) -> SiLU -> Linear(64,64). Maps (B,T) Hz values to (B,T,64) dense embeddings.

### `VoicingEmbedding(embed_dim=32)`
Learned embedding for binary voiced/unvoiced flag. `nn.Embedding(2, embed_dim)` lookup table. When `config.voicing_embed_dim > 1`, this replaces the raw 1-dim scalar voicing input.

### `PhonemeEmbedding(vocab_size=2820, embed_dim=64)`
Base class. `nn.Embedding` lookup table with `padding_idx=0`. Maps (B,T) int64 -> (B,T,64).

### `BlurredPhonemeEmbedding(PhonemeEmbedding)` — extends PhonemeEmbedding
Adds duration-proportional boundary blending. Near phoneme boundaries, produces a weighted average of adjacent phoneme embeddings instead of hard lookup. Same interface: (B,T) int64 -> (B,T,64).

**Algorithm**: Detects boundaries where `ids[:, t] != ids[:, t+1]`, computes per-segment durations, then creates linear blend regions of radius `blend_fraction * min(left_dur, right_dur)` around each boundary. Max blend weight is 0.5 at the boundary itself, tapering to 0 at the edges.

## convnext.py

### `GlobalResponseNorm(dim)`
ConvNeXtV2's inter-channel normalization. Computes per-channel L2 norm across time, normalizes by mean norm, applies learnable scale (`gamma`) and shift (`beta`), plus residual. Initialized to identity (gamma=0, beta=0).

### `ConvNeXtV2Block(dim=512, kernel_size=7, expansion=4, dropout=0.1)`
Single ConvNeXtV2 block: depthwise Conv1d -> LayerNorm -> pointwise expand (4x) -> GELU -> GRN -> pointwise project -> dropout -> residual. ~2.1M params per block.

### `ConvNeXtStack(dim=512, num_blocks=4, kernel_size=7, expansion=4, dropout=0.1)`
Sequential stack of ConvNeXtV2Blocks. With 4 blocks, effective receptive field is ~25 frames (~500ms). Does NOT receive timestep conditioning. ~8.4M params total.

## wavenet.py

Shared WaveNet building blocks used by both architectures. Dilated convolutions with gated (tanh * sigmoid) activations, per-layer timestep conditioning, and accumulated skip connections.

### `WaveNetResidualBlock(channels, cond_channels, skip_channels, kernel_size, dilation, dropout=0.1)`
Single residual block. Pipeline: input dropout -> dilated Conv1d(C->2C, same-padding) -> add per-block conditioning Conv1x1(C_cond->2C) -> split -> `tanh(xa) * sigmoid(xb)` -> 1x1 skip and 1x1 out projections -> residual add scaled by `sqrt(0.5)`. Returns `(x_out, skip)`. Shared by both `WaveNetStack` and `WaveNetDenoiser`.

### `WaveNetStack(hidden_channels=512, ...)`
Used by `VocaloFlowHybrid` for optional pre-processing. Input 1x1 -> N residual blocks with cyclic dilations -> ReLU(skip_sum) -> Conv1x1 -> ReLU -> Conv1x1 -> output. Callers wrap with outer residual `h = h + wavenet_stack(h, c)`. Default 8 blocks at skip=512 ~ 21.7M params.

### `WaveNetDenoiser(residual_channels=256, ...)`
Used by `VocaloFlowWaveNet` as primary backbone. Outputs via skip-sum to configurable width. No `input_conv`. Applies padding mask after each block. Final `output_conv2` is zero-initialized. Default 20 layers, dilation cycle 10. ~13M params at 256ch.

## rope.py

### `precompute_freqs_cis(dim, max_len, theta=10000.0) -> Tensor`
Returns (max_len, dim//2, 2) tensor with precomputed [cos, sin] for each position and frequency.

### `apply_rotary_emb(q, k, freqs_cis) -> (q_rot, k_rot)`
Applies rotation to Q and K tensors (B, H, T, D). Splits into even/odd pairs, rotates using precomputed frequencies, then interleaves back.
