"""PL-BERT feature extractor — loads the frozen pretrained ALBERT model."""

from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AlbertConfig, AlbertModel


_PLBERT_DIR = Path(__file__).parent

_DEFAULT_CONFIG = _PLBERT_DIR / "configs" / "config.yml"
_DEFAULT_CHECKPOINT = _PLBERT_DIR / "checkpoints" / "step_1000000.t7"


class PLBertFeatureExtractor(nn.Module):
    """Frozen PL-BERT model for extracting contextual phoneme embeddings.

    Takes IPA character token IDs (from text_utils.tokenize_ipa) and returns
    768-dim contextual embeddings per token position.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG,
        checkpoint_path: str | Path = _DEFAULT_CHECKPOINT,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        config_path = Path(config_path)
        checkpoint_path = Path(checkpoint_path)

        with open(config_path) as f:
            plbert_config = yaml.safe_load(f)

        albert_config = AlbertConfig(**plbert_config["model_params"])
        self.bert = AlbertModel(albert_config)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["net"]

        cleaned = OrderedDict()
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[len("module."):]
            if name.startswith("encoder."):
                name = name[len("encoder."):]
            cleaned[name] = v

        self.bert.load_state_dict(cleaned, strict=False)
        self.bert.eval()
        for p in self.bert.parameters():
            p.requires_grad = False

        self.to(device)
        self._device = device

    @torch.no_grad()
    def extract(self, token_ids: list[int]) -> torch.Tensor:
        """Run PL-BERT on a sequence of IPA character token IDs.

        Args:
            token_ids: List of integer token IDs (from text_utils.tokenize_ipa).

        Returns:
            (seq_len, 768) tensor of contextual embeddings.
        """
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self._device)
        attention_mask = torch.ones_like(input_ids)
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
