"""modernGENA embedders (AIRI-Institute/GENA_LM).

Implements two encoder-style DNA embedders backed by Hugging Face models:

- ``moderngena-base``  -> ``AIRI-Institute/moderngena-base``
- ``moderngena-large`` -> ``AIRI-Institute/moderngena-large``

The implementation uses standard masked mean pooling over the model's
``last_hidden_state``, matching the embedding style used by other encoder
backends in this project.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import register
from .base import Embedder


class _ModernGENAEmbedder(Embedder):
    """Shared implementation for modernGENA variants."""

    DEFAULT_MODEL_ID: str

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._device_request = device
        self.device: torch.device | None = None
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        self.device = torch.device(
            self._device_request
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # The model docs recommend enabling FlashAttention2 when available.
        model_kwargs: dict[str, object] = {"trust_remote_code": True}
        if importlib.util.find_spec("flash_attn") is not None:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            self.model_id,
            **model_kwargs,
        ).to(self.device)
        self.model.eval()

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            sequences,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens.get("attention_mask")
        if attention_mask is None:
            pad_id = self.tokenizer.pad_token_id
            if pad_id is None:
                raise ValueError(
                    "Tokenizer does not define pad_token_id and did not return "
                    "attention_mask; cannot build a valid padding mask."
                )
            attention_mask = (input_ids != pad_id).long()
        else:
            attention_mask = attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(torch.finfo(hidden.dtype).eps)
        mean_emb = (hidden * mask).sum(dim=1) / denom
        return mean_emb.cpu().float().numpy()


# Canonical names
@register("moderngena-base")
class ModernGENABaseEmbedder(_ModernGENAEmbedder):
    DEFAULT_MODEL_ID = "AIRI-Institute/moderngena-base"


# Canonical names
@register("moderngena-large")
class ModernGENALargeEmbedder(_ModernGENAEmbedder):
    DEFAULT_MODEL_ID = "AIRI-Institute/moderngena-large"
