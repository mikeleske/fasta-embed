"""DNABERT-S embedder (e.g. zhihan1996/DNABERT-S).

Uses simple mean pooling over the first hidden state output.
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from . import register
from .base import Embedder


@register("dnabert")
class DNABERTEmbedder(Embedder):
    DEFAULT_MODEL_ID = "zhihan1996/DNABERT-S"

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
        self.model = AutoModel.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map={"": str(self.device)},
        )

    def embed(self, sequence: str) -> np.ndarray:
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        tokens = self.tokenizer(
            sequences,
            padding=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)
        attention_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            hidden_states = self.model(
                input_ids,
                attention_mask=attention_mask,
            )[0]

        mask = attention_mask.unsqueeze(-1)
        mean_emb = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
        return mean_emb.cpu().numpy()
