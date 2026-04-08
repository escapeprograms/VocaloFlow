# VocaloFlow Model Memory Palace

Neural network architecture for the VocaloFlow conditional flow matching model.

## vocaloflow.py

### `VocaloFlow(config: VocaloFlowConfig)`
Top-level nn.Module. Orchestrates all sub-components.

**Forward signature**: `forward(x_t, t, prior_mel, f0, voicing, phoneme_ids, padding_mask=None) -> Tensor`
- Inputs: x_t (B,T,128), t (B,), prior_mel (B,T,128), f0 (B,T), voicing (B,T), phoneme_ids (B,T) int64, padding_mask (B,T) bool
- Output: (B,T,128) predicted velocity vector

**Forward flow**:
1. `phoneme_embed(phoneme_ids)` → (B,T,256)
2. Concatenate [x_t, prior_mel, f0, voicing, ph_emb] → (B,T,514)
3. `input_proj` Conv1d(514,1024,k=1) → (B,T,1024)
4. `timestep_mlp(t)` → (B,1024) conditioning vector
5. 2x `DiTBlock(h, c, mask)` → (B,T,1024)
6. `output_norm` + `output_proj` Linear(1024,128) → (B,T,128)

## dit_block.py

### `AdaLNZero(hidden_dim)`
Adaptive Layer Normalization with zero initialization. Projects conditioning vector `c` (B, hidden_dim) through SiLU + Linear to produce 6 modulation parameters: gamma1, beta1, alpha1 (for attention), gamma2, beta2, alpha2 (for FFN). Each is (B, 1, hidden_dim).

**Key**: Linear layer is zero-initialized so gamma=0→scale=1, beta=0→shift=0, alpha=0→gate=0. This makes each DiT block start as an identity function, critical for stable training.

### `DiTBlock(hidden_dim=1024, num_heads=16, ffn_dim=4096, max_len=1024)`
Pre-norm transformer block with:
- **Self-attention**: 16 heads, 64 dim/head. Q,K,V via single `qkv` Linear. RoPE applied to Q and K. Uses `F.scaled_dot_product_attention` (FlashAttention when available). Gated residual: `x = x + alpha1 * attn_out`.
- **FFN**: Linear(1024,4096) → GELU → Linear(4096,1024). Gated residual: `x = x + alpha2 * ffn_out`.
- **LayerNorms**: `elementwise_affine=False` — modulation comes from AdaLN instead.
- **Padding mask**: Converted to (B,1,T,T) attention mask for SDPA.
- **RoPE frequencies**: Precomputed and stored as a buffer (`freqs_cis`).

## embeddings.py

### `sinusoidal_timestep_embedding(t, dim, max_period=10000.0) -> Tensor`
Standard sinusoidal embedding for continuous timestep t. Returns (B, dim) with cos/sin pairs.

### `TimestepMLP(hidden_dim=1024, sinusoidal_dim=256)`
Sinusoidal(256) → Linear(256,1024) → SiLU → Linear(1024,1024). Maps scalar timestep t ∈ [0,1] to conditioning vector (B, 1024).

### `PhonemeEmbedding(vocab_size=2820, embed_dim=256)`
`nn.Embedding` lookup table with `padding_idx=0` (PAD token). Maps (B,T) int64 → (B,T,256).

## rope.py

### `precompute_freqs_cis(dim, max_len, theta=10000.0) -> Tensor`
Returns (max_len, dim//2, 2) tensor with precomputed [cos, sin] for each position and frequency.

### `apply_rotary_emb(q, k, freqs_cis) -> (q_rot, k_rot)`
Applies rotation to Q and K tensors (B, H, T, D). Splits into even/odd pairs, rotates using the precomputed frequencies, then interleaves back.
