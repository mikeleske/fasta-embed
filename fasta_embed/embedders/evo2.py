"""Evo 2 embedder (ArcInstitute/evo2).

Wraps the Evo 2 genomic language model (StripedHyena 2 architecture) and
produces fixed-size embeddings via mean pooling over the hidden states of a
chosen intermediate layer.  As reported in the Evo 2 paper, intermediate
layers consistently outperform the final layer for downstream tasks.

Installation
------------
The ``evo2`` package must be installed separately (it is not on PyPI as a
simple ``pip install``):

    # Light install — compatible with any supported GPU (no FP8 required):
    pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2

    # Full install — required for 20B / 40B models (Hopper GPU + FP8):
    conda install -c nvidia cuda-nvcc cuda-cudart-dev
    conda install -c conda-forge transformer-engine-torch=2.3.0
    pip install flash-attn==2.8.0.post2 --no-build-isolation
    pip install evo2

.. note::
    ``evo2_1b_base`` requires FP8 and Transformer Engine (Hopper GPU).
    If you do not have a Hopper GPU, use ``evo2_7b`` instead, which runs
    in bfloat16 on any supported GPU with the light install above.

Choosing a layer
----------------
Pass ``layer_name`` to select which intermediate hidden state to pool.
Layer names follow the pattern ``"blocks.{N}.mlp.l3"``.  The default
(``DEFAULT_LAYER_NAME``) targets roughly 80 % network depth, which is a
well-behaved operating point for biological sequence embeddings.  Adjust
downward for more syntactic / sequence-composition features and upward for
more semantic / functional features.

References
----------
Brixi et al. (2026), "Genome modelling and design across all domains of life
with Evo 2", *Nature*.  https://doi.org/10.1038/s41586-026-10176-5
GitHub: https://github.com/arcinstitute/evo2
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from . import register
from .base import Embedder

log = logging.getLogger(__name__)


@register("evo2")
class Evo2Embedder(Embedder):
    """Evo 2 1B-base embedder registered as ``"evo2"``.

    Parameters
    ----------
    model_id:
        Evo 2 checkpoint name passed to ``Evo2(model_id)``.
        Defaults to ``"evo2_1b_base"``.  Other valid names include
        ``"evo2_7b"``, ``"evo2_7b_base"``, ``"evo2_20b"``, ``"evo2_40b"``.
    device:
        PyTorch device string (e.g. ``"cuda:0"`` or ``"cpu"``).
        Auto-detected when ``None``.
    layer_name:
        Name of the internal block whose hidden state is pooled to form the
        embedding.  Follows Evo 2's ``layer_names`` API.  Defaults to
        ``DEFAULT_LAYER_NAME``.
    """

    DEFAULT_MODEL_ID = "evo2_1b_base"

    # Roughly 80 % depth of the 1B model (~24 transformer blocks).
    # Adjust if you switch to a larger checkpoint (e.g. "blocks.28.mlp.l3"
    # for the 7B model which has 32 blocks).
    DEFAULT_LAYER_NAME = "blocks.20.mlp.l3"

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        layer_name: str | None = None,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL_ID
        self._device_request = device
        self.device: torch.device | None = None
        self.layer_name: str = layer_name or self.DEFAULT_LAYER_NAME
        self._model = None

    # ------------------------------------------------------------------
    # Embedder interface
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Download (if needed) and initialise the Evo 2 model.

        ``evo2`` is imported here rather than at module level so that the
        package remains importable even when ``evo2`` is not installed —
        allowing the registry to list all other embedders normally.
        """
        try:
            from evo2 import Evo2  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "The 'evo2' package is required for the Evo2 embedder.\n"
                "Install it with:\n"
                "  pip install flash-attn==2.8.0.post2 --no-build-isolation\n"
                "  pip install evo2\n"
                "See https://github.com/arcinstitute/evo2 for full details."
            ) from exc

        self.device = torch.device(
            self._device_request
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        log.info("Loading Evo 2 checkpoint '%s' ...", self.model_id)
        self._model = Evo2(self.model_id)
        log.info(
            "Evo 2 model loaded. Embedding layer: '%s'", self.layer_name
        )

    def embed(self, sequence: str) -> np.ndarray:
        """Return a 1-D embedding vector for a single *sequence*."""
        return self.embed_batch([sequence])[0]

    def embed_batch(self, sequences: list[str]) -> np.ndarray:
        """Return a 2-D array of shape ``(len(sequences), hidden_dim)``.

        Each sequence is tokenized individually and forwarded through Evo 2
        with ``return_embeddings=True``.  The hidden states at
        ``self.layer_name`` are mean-pooled over the sequence length
        dimension to yield one vector per sequence.

        Processing is sequential rather than truly batched because Evo 2
        uses a character-level tokenizer that produces variable-length
        token tensors; unified batching would require padding logic that
        is not exposed in the public API.
        """
        device_str = str(self.device)
        vectors: list[np.ndarray] = []

        for seq in sequences:
            input_ids = torch.tensor(
                self._model.tokenizer.tokenize(seq),
                dtype=torch.int,
            ).unsqueeze(0).to(device_str)

            with torch.no_grad():
                _, layer_embeddings = self._model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.layer_name],
                )

            # layer_embeddings: dict[str, Tensor]  shape (1, seq_len, D)
            hidden = layer_embeddings[self.layer_name]  # (1, L, D)
            vec = hidden.squeeze(0).mean(dim=0).cpu().float().numpy()
            vectors.append(vec)

        return np.vstack(vectors)
