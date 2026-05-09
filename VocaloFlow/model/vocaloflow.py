"""VocaloFlow: Base class for conditional flow matching singing voice models.

Shared conditioning encoder (phoneme, F0, voicing, timestep, speaker) and
template-method ``forward()`` that delegates to ``_backbone_forward()``
implemented by each subclass.

Subclasses:
  - ``VocaloFlowHybrid``  (vocaloflow_hybrid.py) — DiT backbone + optional
    ConvNeXt / WaveNet pre-processing.
  - ``VocaloFlowWaveNet`` (vocaloflow_wavenet.py) — WaveNet backbone + optional
    DiT refinement.
"""

import sys
import os

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from configs.default import VocaloFlowConfig
from model.embeddings import (
    TimestepMLP,
    PhonemeEmbedding,
    BlurredPhonemeEmbedding,
    F0Embedding,
    VoicingEmbedding,
)


class VocaloFlow(nn.Module):
    """Base conditional flow matching model for singing voice enhancement.

    Owns all shared conditioning modules and provides a template-method
    ``forward()`` that calls ``_encode_conditioning()`` followed by
    ``_backbone_forward()``.  Subclasses only need to implement
    ``_backbone_forward`` and register their backbone-specific modules.

    Args:
        config: VocaloFlowConfig with all hyperparameters.
        hidden_dim: Width of the projected conditioning / backbone hidden state.
            Hybrid uses ``config.hidden_dim`` (512); WaveNet uses
            ``config.wavenet_pure_residual_channels`` (256).
    """

    def __init__(self, config: VocaloFlowConfig, hidden_dim: int) -> None:
        super().__init__()
        self.config = config
        self.hidden_dim = hidden_dim

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
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # ── Timestep conditioning MLP ────────────────────────────────────
        self.timestep_mlp = TimestepMLP(hidden_dim=hidden_dim, sinusoidal_dim=hidden_dim // 2)

        # ── Speaker embedding conditioning ───────────────────────────────
        self.use_speaker_embedding = config.use_speaker_embedding
        if config.use_speaker_embedding:
            self.speaker_proj = nn.Linear(config.speaker_embedding_dim, hidden_dim)
            nn.init.zeros_(self.speaker_proj.weight)
            nn.init.zeros_(self.speaker_proj.bias)

    # ── Shared conditioning encoder ──────────────────────────────────────

    def _encode_conditioning(
        self,
        x_t: Tensor,
        t: Tensor,
        prior_mel: Tensor,
        f0: Tensor,
        voicing: Tensor,
        phoneme_ids: Tensor,
        plbert_features: Tensor | None = None,
        speaker_embedding: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode all conditioning inputs into hidden state and conditioning vector.

        Args:
            x_t:               (B, T, 128) interpolated mel state.
            t:                 (B,) flow timestep in [0, 1].
            prior_mel:         (B, T, 128) Vocaloid prior mel (x_0 conditioning).
            f0:                (B, T) F0 contour (0 for unvoiced).
            voicing:           (B, T) voiced/unvoiced binary flag.
            phoneme_ids:       (B, T) resolved phoneme token IDs.
            plbert_features:   (B, T, 768) frozen PL-BERT embeddings (optional).
            speaker_embedding: (B, 192) ECAPA-TDNN speaker embedding (optional).

        Returns:
            h: (B, T, hidden_dim) projected conditioning tensor.
            c: (B, hidden_dim) timestep + speaker conditioning vector.
        """
        # 1. Embed phonemes
        if self.use_plbert and plbert_features is not None:
            ph_emb = self.plbert_proj(plbert_features)
        else:
            ph_emb = self.phoneme_embed(phoneme_ids)

        # 2. Embed F0
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
            x_t_normed,
            prior_normed,
            f0_normed,
            ph_normed,
            voicing_emb,
        ], dim=-1)

        # 6. Input projection
        h = self.input_proj(cond)

        # 7. Timestep conditioning
        c = self.timestep_mlp(t)

        # 7b. Speaker embedding conditioning
        if self.use_speaker_embedding and speaker_embedding is not None:
            c = c + self.speaker_proj(speaker_embedding)

        return h, c

    # ── Template method ──────────────────────────────────────────────────

    def _backbone_forward(
        self,
        h: Tensor,
        c: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Architecture-specific backbone. Subclasses must override.

        Args:
            h: (B, T, hidden_dim) projected conditioning tensor.
            c: (B, hidden_dim) timestep + speaker conditioning vector.
            padding_mask: (B, T) bool, True = valid frame.

        Returns:
            (B, T, mel_channels) predicted velocity vector.
        """
        raise NotImplementedError

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
        h, c = self._encode_conditioning(
            x_t, t, prior_mel, f0, voicing, phoneme_ids,
            plbert_features, speaker_embedding,
        )
        return self._backbone_forward(h, c, padding_mask)
