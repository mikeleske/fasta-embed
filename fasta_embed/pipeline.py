"""Orchestration: iterate sequences, embed in batches, save to disk."""

from __future__ import annotations

import logging
from tqdm import tqdm

from .config import EmbedConfig
from .embedders.base import Embedder
from .io import EmbeddingWriter, SequenceBatches, iter_sequences  # noqa: F401

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
    log.info(
        "Found %d sequences (%d batches of %d)",
        batches.total_sequences,
        len(batches),
        config.inference_batch_size,
    )

    vectorize(embedder, batches, config.output_file)


def vectorize(
    embedder: Embedder,
    batches: SequenceBatches,
    output_file: str,
) -> None:
    """Consume *batches* of sequences, embed each batch, and stream to disk.

    Parameters
    ----------
    embedder:
        A loaded :class:`Embedder` instance.
    batches:
        Iterator yielding lists of sequence strings (one list per inference
        batch, already sized by ``iter_sequences``).
    output_file:
        Path for the ``.npy`` output.
    """
    writer = EmbeddingWriter(output_file)

    for batch in tqdm(batches, desc="Embedding"):
        embs = embedder.embed_batch(batch)
        writer.append(embs)

    writer.finalize()
    log.info(
        "Done. %d embeddings saved to %s", writer.count, output_file,
    )
