"""Sequence I/O (FASTA / CSV loading) and embedding persistence."""

from __future__ import annotations

import gzip
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

from .bio import get_region


# ------------------------------------------------------------------
# Format detection
# ------------------------------------------------------------------

_FASTA_EXTENSIONS = {".fasta", ".fa", ".fna", ".fas"}
_CSV_EXTENSIONS = {".csv", ".tsv", ".txt"}


def _infer_format(path: Path) -> str:
    """Guess file format from the extension (stripping .gz first)."""
    suffixes = path.suffixes
    ext = (
        suffixes[-2].lower()
        if len(suffixes) >= 2 and suffixes[-1].lower() == ".gz"
        else path.suffix.lower()
    )

    if ext in _FASTA_EXTENSIONS:
        return "fasta"
    if ext in _CSV_EXTENSIONS:
        return "csv"
    raise ValueError(
        f"Cannot infer format from '{path.name}'. "
        f"Use --input-format to specify 'fasta' or 'csv'."
    )


# ------------------------------------------------------------------
# Lazy single-sequence iterators
# ------------------------------------------------------------------

def _iter_fasta(path: Path) -> Iterator[str]:
    """Yield one sequence string at a time from a (possibly gzipped) FASTA."""
    is_gzipped = (
        path.suffix == ".gz"
        or path.suffixes[-2:] == [".fasta", ".gz"]
    )
    opener = gzip.open if is_gzipped else open
    with opener(path, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            yield str(rec.seq)


def _iter_csv(path: Path, sep: str, column: str) -> Iterator[str]:
    """Yield one sequence string at a time from a CSV/TSV, using chunked
    reading to keep memory usage low."""
    for chunk in pd.read_csv(path, sep=sep, usecols=[column], chunksize=10_000):
        yield from chunk[column]


# ------------------------------------------------------------------
# Batched sequence iterator (main entry point for the pipeline)
# ------------------------------------------------------------------

def iter_sequences(
    path: str | Path,
    fmt: str | None = None,
    *,
    batch_size: int = 16,
    csv_separator: str = "\t",
    sequence_column: str = "Seq",
    region: str | None = None,
) -> Iterator[list[str]]:
    """Yield batches of sequences from *path*.

    Handles format detection, column selection, and optional 16S region
    extraction.  Each yielded item is a list of up to *batch_size* strings.

    Parameters
    ----------
    path:
        File to read (FASTA, gzipped FASTA, or CSV/TSV).
    fmt:
        ``"fasta"`` or ``"csv"``.  Inferred from the file extension when
        ``None``.
    batch_size:
        Maximum number of sequences per yielded batch.
    csv_separator:
        Delimiter used when *fmt* is ``"csv"``.
    sequence_column:
        Column containing sequences (CSV only).
    region:
        Optional 16S region to extract (e.g. ``"V3V4"``).
    """
    path = Path(path)
    fmt = fmt or _infer_format(path)

    if fmt == "fasta":
        seqs = _iter_fasta(path)
    elif fmt == "csv":
        seqs = _iter_csv(path, csv_separator, sequence_column)
    else:
        raise ValueError(
            f"Unsupported input format: '{fmt}' (use 'fasta' or 'csv')"
        )

    if region:
        seqs = (get_region(region, s) for s in seqs)

    batch: list[str] = []
    for seq in seqs:
        batch.append(seq)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ------------------------------------------------------------------
# Eager DataFrame loader (kept for programmatic / notebook use)
# ------------------------------------------------------------------

def load_sequences(
    path: str | Path,
    fmt: str | None = None,
    *,
    csv_separator: str = "\t",
    sequence_column: str = "Seq",
) -> pd.DataFrame:
    """Read sequences from *path* and return a DataFrame.

    Parameters
    ----------
    path:
        File to read (FASTA, gzipped FASTA, or CSV/TSV).
    fmt:
        ``"fasta"`` or ``"csv"``.  Inferred from the file extension when
        ``None``.
    csv_separator:
        Delimiter used when *fmt* is ``"csv"``.
    sequence_column:
        Ensures the returned DataFrame contains this column.
        For FASTA inputs the column is always ``"Seq"``.

    Returns
    -------
    pd.DataFrame
        Must contain at least the *sequence_column* column.
    """
    path = Path(path)
    fmt = fmt or _infer_format(path)

    if fmt == "fasta":
        return _parse_fasta(path)
    if fmt == "csv":
        df = pd.read_csv(path, sep=csv_separator)
        if sequence_column not in df.columns:
            raise ValueError(
                f"Column '{sequence_column}' not found in {path}. "
                f"Available: {list(df.columns)}"
            )
        return df

    raise ValueError(
        f"Unsupported input format: '{fmt}' (use 'fasta' or 'csv')"
    )


def _parse_fasta(path: Path) -> pd.DataFrame:
    """Parse a (possibly gzipped) FASTA file into a DataFrame with columns
    ``ID``, ``SeqLen``, ``Seq``."""
    is_gzipped = (
        path.suffix == ".gz"
        or path.suffixes[-2:] == [".fasta", ".gz"]
    )
    opener = gzip.open if is_gzipped else open

    rows: list[dict] = []
    with opener(path, "rt") as handle:
        for rec in SeqIO.parse(handle, "fasta"):
            rows.append(
                {
                    "ID": str(rec.description).split()[0],
                    "SeqLen": len(rec.seq),
                    "Seq": str(rec.seq),
                }
            )
    return pd.DataFrame(rows, columns=["ID", "SeqLen", "Seq"])


# ------------------------------------------------------------------
# Embedding persistence
# ------------------------------------------------------------------

def save_embeddings_batch(
    vectors: np.ndarray,
    file_path: str | Path,
) -> np.ndarray:
    """Append *vectors* to the numpy file at *file_path*, creating it if
    necessary.  Returns the full (concatenated) array on disk."""
    file_path = Path(file_path)
    if file_path.exists():
        existing = np.load(file_path)
        combined = np.concatenate((existing, vectors), axis=0)
    else:
        combined = vectors
    np.save(file_path, combined)
    return combined
