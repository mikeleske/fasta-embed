"""Nucleotide Transformer v3 embedder (e.g. InstaDeepAI/NTv3_8M_pre).

Uses masked-LM model class with attention-mask-weighted mean pooling over the
last hidden state.  Supports efficient batched inference via ``embed_batch``.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from . import register
from .base import Embedder


@register("ntv3")
class NTv3Embedder(Embedder):
    DEFAULT_MODEL_ID = "InstaDeepAI/NTv3_100M_pre"

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
            self._device_request or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map={"": str(self.device)},
        )

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            sequences,
            add_special_tokens=False,
            padding=True,
            pad_to_multiple_of=128,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = (
            (input_ids != self.tokenizer.pad_token_id).long().to(self.device)
        )

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        hidden = outputs.hidden_states[-1]
        mask = attention_mask.unsqueeze(-1)
        mean_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
        return mean_emb.cpu().numpy()
