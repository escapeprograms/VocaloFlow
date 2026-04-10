# VocaloFlow Model Memory Palace

Neural network architecture for the VocaloFlow conditional flow matching model (v2 — deeper/narrower with regularization).

## vocaloflow.py

### `VocaloFlow(config: VocaloFlowConfig)`
Top-level nn.Module. Orchestrates all sub-components.

**Forward signature**: `forward(x_t, t, prior_mel, f0, voicing, phoneme_ids, padding_mask=None) -> Tensor`
- Inputs: x_t (B,T,128), t (B,), prior_mel (B,T,128), f0 (B,T), voicing (B,T), phoneme_ids (B,T) int64, padding_mask (B,T) bool
- Output: (B,T,128) predicted velocity vector

**Forward flow**:
1. `phoneme_embed(phoneme_ids)` → (B,T,64)
2. `f0_embed(f0)` → (B,T,64) via learned MLP
3. Per-stream LayerNorm on x_t (128), prior_mel (128), f0_emb (64), ph_emb (64). vuv (1) passed raw.
4. Concatenate all → (B,T,385)
5. `input_proj` Linear(385,512) → (B,T,512)
6. `timestep_mlp(t)` → (B,512) conditioning vector
7. 6x `DiTBlock(h, c, mask)` → (B,T,512)
8. `output_norm` + `output_proj` Linear(512,128) → (B,T,128)

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

### `PhonemeEmbedding(vocab_size=2820, embed_dim=64)`
`nn.Embedding` lookup table with `padding_idx=0` (PAD token). Maps (B,T) int64 → (B,T,64). Reduced from 256 to 64 so phoneme identity doesn't dominate the input over the prior mel signal.

## rope.py

### `precompute_freqs_cis(dim, max_len, theta=10000.0) -> Tensor`
Returns (max_len, dim//2, 2) tensor with precomputed [cos, sin] for each position and frequency.

### `apply_rotary_emb(q, k, freqs_cis) -> (q_rot, k_rot)`
Applies rotation to Q and K tensors (B, H, T, D). Splits into even/odd pairs, rotates using the precomputed frequencies, then interleaves back.
