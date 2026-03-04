"""Sequence I/O (FASTA / CSV loading) and embedding persistence."""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO


# ------------------------------------------------------------------
# Sequence loading
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

    raise ValueError(f"Unsupported input format: '{fmt}' (use 'fasta' or 'csv')")  # noqa: E501


def _parse_fasta(path: Path) -> pd.DataFrame:
    """Parse a (possibly gzipped) FASTA file into a DataFrame with columns
    ``ID``, ``SeqLen``, ``Seq``."""
    is_gzipped = path.suffix == ".gz" or path.suffixes[-2:] == [".fasta", ".gz"]  # noqa: E501
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

def save_embeddings_batch(vectors: np.ndarray, file_path: str | Path) -> np.ndarray:  # noqa: E501
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
