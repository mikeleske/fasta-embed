from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Embedder(ABC):
    """Backend-agnostic interface for sequence embedding strategies.

    Subclasses must implement ``load`` (one-time resource initialization) and
    ``embed`` (single-sequence embedding).  ``embed_batch`` has a default
    implementation that iterates over sequences, but concrete embedders should
    override it when the backend supports efficient batched inference.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights / open connections.  Called once before any
        embedding work begins."""

    @abstractmethod
    def embed(self, sequence: str) -> np.ndarray:
        """Return a 1-D embedding vector for *sequence*."""

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Return a 2-D array of shape ``(len(sequences), embed_dim)``.

        The default loops over ``embed`` -- override for backends that
        benefit from true batching (GPU models, API calls, etc.).
        """
        return np.vstack([self.embed(seq) for seq in sequences])
