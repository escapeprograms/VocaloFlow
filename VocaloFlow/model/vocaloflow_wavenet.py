"""VocaloFlow Pure WaveNet: Flow matching denoiser with optional DiT refinement.

WaveNet backbone optionally followed by DiT transformer blocks for global
refinement.  Same forward signature as VocaloFlow for drop-in compatibility
with the training loop, loss computation, and inference pipeline.

Selected via ``config.architecture = "wavenet_pure"``.  DiT blocks are
controlled by ``config.wavenet_pure_num_dit_blocks`` (0 = pure WaveNet).
"""

import sys
import os

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.dit_block import DiTBlock
from model.wavenet import WaveNetDenoiser
from model.embeddings import (
    TimestepMLP,
    PhonemeEmbedding,
    BlurredPhonemeEmbedding,
    F0Embedding,
    VoicingEmbedding,
)


class VocaloFlowPureWaveNet(nn.Module):
    """Pure WaveNet conditional flow matching model with optional DiT refinement.

    Same forward signature as VocaloFlow.  Uses a WaveNetDenoiser backbone,
    optionally followed by DiT transformer blocks for global attention.

    Architecture:
      1. Phoneme embedding: (B, T) -> (B, T, 64) [PL-BERT or learned lookup]
      2. F0 embedding: (B, T) -> (B, T, 64)
      3. Per-stream LayerNorm on x_t (128), prior_mel (128), f0 (64), ph (64)
      4. Concatenate + voicing: (B, T, input_dim)
      5. Input projection: Linear(input_dim, rc)
      6. Timestep conditioning: sinusoidal -> MLP -> (B, rc)
      7. Optional speaker embedding: additive to conditioning vector
      8. WaveNetDenoiser: residual blocks with skip-sum
      9. Optional DiT blocks: transformer with AdaLN-Zero, RoPE (if num_dit > 0)
     10. Output projection: LayerNorm -> Linear(rc, 128) (if DiT) or direct (if pure)

    Args:
        config: VocaloFlowConfig with architecture="wavenet_pure".
    """

    def __init__(self, config: VocaloFlowConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = VocaloFlowConfig()
        self.config = config

        rc = config.wavenet_pure_residual_channels

        # ── Phoneme conditioning ─────────────────────────────────────────
        self.use_plbert = config.use_plbert
        if config.use_plbert:
            self.plbert_proj = nn.Linear(config.plbert_feature_dim, config.plbert_proj_dim)

        if config.phoneme_blur_enabled:
            self.phoneme_embed = BlurredPhonemeEmbedding(
                vocab_size=config.phoneme_vocab_size,
                embed_dim=config.phoneme_embed_dim,
                blend_fraction=config.phoneme_blend_fraction,
            )
        else:
            self.phoneme_embed = PhonemeEmbedding(
                vocab_size=config.phoneme_vocab_size,
                embed_dim=config.phoneme_embed_dim,
            )

        # ── F0 and voicing ──────────────────────────────────────────────
        self.f0_embed = F0Embedding(embed_dim=config.f0_embed_dim)

        self.voicing_embed_dim = config.voicing_embed_dim
        if config.voicing_embed_dim > 1:
            self.voicing_embed = VoicingEmbedding(embed_dim=config.voicing_embed_dim)
            self.norm_voicing = nn.LayerNorm(config.voicing_embed_dim)

        # ── Per-stream normalization ─────────────────────────────────────
        self.norm_xt = nn.LayerNorm(config.mel_channels)
        self.norm_prior = nn.LayerNorm(config.mel_channels)
        self.norm_f0 = nn.LayerNorm(config.f0_embed_dim)
        self.norm_ph = nn.LayerNorm(config.phoneme_embed_dim)

        # ── Input projection ─────────────────────────────────────────────
        input_dim = (config.mel_channels * 2
                     + config.f0_embed_dim
                     + config.phoneme_embed_dim
                     + config.voicing_embed_dim)
        self.input_proj = nn.Linear(input_dim, rc)

        # ── Timestep conditioning MLP ────────────────────────────────────
        self.timestep_mlp = TimestepMLP(hidden_dim=rc)

        # ── Speaker embedding conditioning ───────────────────────────────
        self.use_speaker_embedding = config.use_speaker_embedding
        if config.use_speaker_embedding:
            self.speaker_proj = nn.Linear(config.speaker_embedding_dim, rc)
            nn.init.zeros_(self.speaker_proj.weight)
            nn.init.zeros_(self.speaker_proj.bias)

        # ── WaveNet backbone ─────────────────────────────────────────────
        self.has_dit = config.wavenet_pure_num_dit_blocks > 0
        wavenet_output = rc if self.has_dit else config.mel_channels

        self.denoiser = WaveNetDenoiser(
            residual_channels=rc,
            cond_channels=rc,
            skip_channels=config.wavenet_pure_skip_channels,
            mel_channels=config.mel_channels,
            output_channels=wavenet_output,
            kernel_size=config.wavenet_pure_kernel_size,
            n_layers=config.wavenet_pure_num_layers,
            dilation_cycle=config.wavenet_pure_dilation_cycle,
            dropout=config.wavenet_pure_dropout,
        )

        # ── DiT refinement blocks (optional) ────────────────────────────
        if self.has_dit:
            self.dit_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_dim=rc,
                    num_heads=config.num_heads,
                    ffn_dim=config.ffn_dim,
                    max_len=config.max_seq_len * 2,
                    dropout=config.dropout,
                )
                for _ in range(config.wavenet_pure_num_dit_blocks)
            ])
            self.output_norm = nn.LayerNorm(rc)
            self.output_proj = nn.Linear(rc, config.mel_channels)
        else:
            self.dit_blocks = None
            self.output_norm = None
            self.output_proj = None

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        prior_mel: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        padding_mask: Tensor | None = None,
        plbert_features: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
    ) -> Tensor:
        """Predict the velocity field v_theta.

        Args:
            x_t:               (B, T, 128) interpolated mel state.
            t:                 (B,) flow timestep in [0, 1].
            prior_mel:         (B, T, 128) Vocaloid prior mel (x_0 conditioning).
            f0:                (B, T) F0 contour (0 for unvoiced).
            voicing:           (B, T) voiced/unvoiced binary flag.
            phoneme_ids:       (B, T) resolved phoneme token IDs.
            padding_mask:      (B, T) bool, True = valid frame.
            plbert_features:   (B, T, 768) frozen PL-BERT embeddings (optional).
            speaker_embedding: (B, 192) ECAPA-TDNN speaker embedding (optional).

        Returns:
            (B, T, 128) predicted velocity vector.
        """
        # 1. Embed phonemes: (B, T) -> (B, T, 64)
        if self.use_plbert and plbert_features is not None:
            ph_emb = self.plbert_proj(plbert_features)
        else:
            ph_emb = self.phoneme_embed(phoneme_ids)

        # 2. Embed F0: (B, T) -> (B, T, 64)
        f0_emb = self.f0_embed(f0)

        # 3. Per-stream normalization
        x_t_normed = self.norm_xt(x_t)
        prior_normed = self.norm_prior(prior_mel)
        f0_normed = self.norm_f0(f0_emb)
        ph_normed = self.norm_ph(ph_emb)

        # 4. Voicing: learned embedding or raw scalar
        if self.voicing_embed_dim > 1:
            voicing_emb = self.norm_voicing(self.voicing_embed(voicing))
        else:
            voicing_emb = voicing.unsqueeze(-1)

        # 5. Concatenate all inputs
        cond = torch.cat([
            x_t_normed,                         # (B, T, 128)
            prior_normed,                        # (B, T, 128)
            f0_normed,                           # (B, T, 64)
            ph_normed,                           # (B, T, 64)
            voicing_emb,                         # (B, T, voicing_embed_dim)
        ], dim=-1)

        # 6. Input projection
        h = self.input_proj(cond)

        # 7. Timestep conditioning: (B,) -> (B, rc)
        c = self.timestep_mlp(t)

        # 7b. Speaker embedding conditioning (additive, zero-init preserves pretrained behavior)
        if self.use_speaker_embedding and speaker_embedding is not None:
            c = c + self.speaker_proj(speaker_embedding)

        # 8. WaveNet backbone
        h = self.denoiser(h, c, padding_mask)

        # 9. DiT refinement blocks (optional)
        if self.dit_blocks is not None:
            for block in self.dit_blocks:
                h = block(h, c, padding_mask)

            # 10. Output projection: LayerNorm -> Linear -> (B, T, 128)
            v = self.output_proj(self.output_norm(h))
            return v

        return h
