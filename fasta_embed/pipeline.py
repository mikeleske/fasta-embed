"""Orchestration: load sequences, optionally extract regions, vectorize, save."""  # noqa: E501

from __future__ import annotations

import logging

import numpy as np
from tqdm import tqdm

from .bio import get_region
from .config import EmbedConfig
from .embedders.base import Embedder
from .io import load_sequences, save_embeddings_batch

log = logging.getLogger(__name__)


def run(embedder: Embedder, config: EmbedConfig) -> None:
    """Execute a full embed-and-save pipeline described by *config*."""

    log.info(
        "Loading sequences from %s (format=%s)",
        config.input_file,
        config.input_format
    )  # noqa: E501
    df = load_sequences(
        config.input_file,
        fmt=config.input_format,
        csv_separator=config.csv_separator,
        sequence_column=config.sequence_column,
    )
    log.info("Loaded %d sequences", len(df))

    column = config.sequence_column
    if config.region:
        log.info("Extracting region %s", config.region)
        df[config.region] = df[column].apply(
            lambda seq: get_region(config.region, seq)
        )
        column = config.region

    vectorize(
        embedder,
        sequences=df[column].tolist(),
        output_file=config.output_file,
        inference_batch_size=config.inference_batch_size,
        save_batch_size=config.save_batch_size,
    )


def vectorize(
    embedder: Embedder,
    sequences: list[str],
    output_file: str,
    inference_batch_size: int = 16,
    save_batch_size: int = 10_000,
) -> None:
    """Generate embeddings in batches and flush to disk periodically.

    Parameters
    ----------
    embedder:
        A loaded :class:`Embedder` instance.
    sequences:
        Flat list of DNA strings.
    output_file:
        Path for the ``.npy`` output.
    inference_batch_size:
        Sequences per model forward pass.
    save_batch_size:
        Embeddings to accumulate before flushing to disk.
    """
    vectors: list[np.ndarray] = []
    accumulated = 0

    for start in tqdm(
        range(0, len(sequences), inference_batch_size),
        desc="Embedding",
    ):
        batch = sequences[start: start + inference_batch_size]
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
