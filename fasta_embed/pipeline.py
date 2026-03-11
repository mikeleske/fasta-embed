"""Orchestration: iterate sequences, embed in batches, save to disk."""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
from tqdm import tqdm

from .config import EmbedConfig
from .embedders.base import Embedder
from .io import iter_sequences, save_embeddings_batch

log = logging.getLogger(__name__)


def run(embedder: Embedder, config: EmbedConfig) -> None:
    """Execute a full embed-and-save pipeline described by *config*."""
    log.info("Reading sequences from %s", config.input_file)
    if config.region:
        log.info("Extracting region %s", config.region)

    batches = iter_sequences(
        config.input_file,
        fmt=config.input_format,
        batch_size=config.inference_batch_size,
        csv_separator=config.csv_separator,
        sequence_column=config.sequence_column,
        region=config.region,
    )

    vectorize(embedder, batches, config.output_file, config.save_batch_size)


def vectorize(
    embedder: Embedder,
    batches: Iterator[list[str]],
    output_file: str,
    save_batch_size: int = 10_000,
) -> None:
    """Consume *batches* of sequences, embed each batch, and flush to disk.

    Parameters
    ----------
    embedder:
        A loaded :class:`Embedder` instance.
    batches:
        Iterator yielding lists of sequence strings (one list per inference
        batch, already sized by ``iter_sequences``).
    output_file:
        Path for the ``.npy`` output.
    save_batch_size:
        Embeddings to accumulate before flushing to disk.
    """
    vectors: list[np.ndarray] = []
    accumulated = 0

    for batch in tqdm(batches, desc="Embedding"):
        embs = embedder.embed_batch(batch)
        vectors.append(embs)
        accumulated += embs.shape[0]

        if accumulated >= save_batch_size:
            save_embeddings_batch(np.vstack(vectors), output_file)
            vectors = []
            accumulated = 0

    if vectors:
        save_embeddings_batch(np.vstack(vectors), output_file)

    log.info("Done. Embeddings saved to %s", output_file)
