"""Sequence I/O (FASTA / CSV loading) and embedding persistence."""

from __future__ import annotations

import gzip
import math
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
# Fast sequence counting
# ------------------------------------------------------------------

def _count_fasta(path: Path) -> int:
    """Count records in a (possibly gzipped) FASTA by counting '>' lines."""
    is_gzipped = (
        path.suffix == ".gz"
        or path.suffixes[-2:] == [".fasta", ".gz"]
    )
    opener = gzip.open if is_gzipped else open
    count = 0
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                count += 1
    return count


def _count_csv(path: Path) -> int:
    """Count data rows in a (possibly gzipped) CSV by counting lines minus
    the header."""
    is_gzipped = path.suffix == ".gz"
    opener = gzip.open if is_gzipped else open
    count = -1  # exclude header
    with opener(path, "rt") as fh:
        for _ in fh:
            count += 1
    return max(count, 0)


# ------------------------------------------------------------------
# Batched sequence iterator (main entry point for the pipeline)
# ------------------------------------------------------------------

class SequenceBatches:
    """Iterable that yields batches of sequences and exposes ``__len__``
    (total number of batches) so ``tqdm`` can display a proper progress bar.

    Prefer using :func:`iter_sequences` to construct instances.
    """

    def __init__(
        self,
        path: Path,
        fmt: str,
        batch_size: int,
        csv_separator: str,
        sequence_column: str,
        region: str | None,
        total_sequences: int,
    ) -> None:
        self._path = path
        self._fmt = fmt
        self._batch_size = batch_size
        self._csv_separator = csv_separator
        self._sequence_column = sequence_column
        self._region = region
        self.total_sequences = total_sequences

    def __len__(self) -> int:
        return math.ceil(self.total_sequences / self._batch_size)

    def __iter__(self) -> Iterator[list[str]]:
        if self._fmt == "fasta":
            seqs = _iter_fasta(self._path)
        else:
            seqs = _iter_csv(
                self._path, self._csv_separator, self._sequence_column,
            )

        if self._region:
            seqs = (get_region(self._region, s) for s in seqs)

        batch: list[str] = []
        for seq in seqs:
            batch.append(seq)
            if len(batch) >= self._batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def iter_sequences(
    path: str | Path,
    fmt: str | None = None,
    *,
    batch_size: int = 16,
    csv_separator: str = "\t",
    sequence_column: str = "Seq",
    region: str | None = None,
) -> SequenceBatches:
    """Return a :class:`SequenceBatches` iterable over *path*.

    Handles format detection, column selection, and optional 16S region
    extraction.  Each yielded item is a list of up to *batch_size* strings.
    ``len()`` returns the total number of batches.

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
        total = _count_fasta(path)
    elif fmt == "csv":
        total = _count_csv(path)
    else:
        raise ValueError(
            f"Unsupported input format: '{fmt}' (use 'fasta' or 'csv')"
        )

    return SequenceBatches(
        path=path,
        fmt=fmt,
        batch_size=batch_size,
        csv_separator=csv_separator,
        sequence_column=sequence_column,
        region=region,
        total_sequences=total,
    )


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

class EmbeddingWriter:
    """Efficiently accumulate embeddings by appending raw bytes to a
    temporary binary file.  Each ``append`` call is O(batch) regardless
    of how many embeddings have already been written.

    Call :meth:`finalize` once at the end to produce the ``.npy`` output.

    Usage::

        writer = EmbeddingWriter("embeddings.npy")
        writer.append(batch_1)
        writer.append(batch_2)
        writer.finalize()
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self._raw_path = self.output_path.with_suffix(".bin")
        self._rows = 0
        self._embed_dim: tuple[int, ...] | None = None
        self._dtype: np.dtype | None = None

    @property
    def count(self) -> int:
        """Total number of embedding rows written so far."""
        return self._rows

    def append(self, vectors: np.ndarray) -> int:
        """Append *vectors* to the backing binary file.

        Returns the cumulative number of rows written.
        """
        if self._dtype is None:
            self._dtype = vectors.dtype
            self._embed_dim = vectors.shape[1:]
        with open(self._raw_path, "ab") as fh:
            fh.write(np.ascontiguousarray(vectors).tobytes())
        self._rows += vectors.shape[0]
        return self._rows

    def finalize(self) -> None:
        """Convert the raw binary file into the final ``.npy`` and clean up."""
        if self._rows == 0:
            return
        shape = (self._rows, *self._embed_dim)
        raw = np.memmap(
            self._raw_path, dtype=self._dtype, mode="r", shape=shape,
        )
        np.save(self.output_path, raw)
        del raw
        self._raw_path.unlink()
