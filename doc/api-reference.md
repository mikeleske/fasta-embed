# API Reference

Complete module-level documentation for every public class, function, and constant in the `fasta_embed` package.

---

## `fasta_embed.__main__`

CLI entry point invoked via `python -m fasta_embed`.

### Functions

#### `_build_parser() → argparse.ArgumentParser`

Constructs the argument parser with all CLI flags. See [Configuration](configuration.md) for the full flag table.

#### `main(argv: list[str] | None = None) → None`

Top-level orchestrator:

1. Parses CLI arguments.
2. Configures logging (`DEBUG` if `--verbose`, else `INFO`).
3. If `--list-embedders`: prints registered names and exits.
4. Builds `EmbedConfig` from YAML (if `--config`) or defaults, then merges CLI overrides.
5. Creates and loads the embedder via the registry.
6. Calls `pipeline.run(embedder, config)`.

---

## `fasta_embed.config`

Dataclass-based configuration with YAML and CLI override support.

### Classes

#### `EmbedConfig`

```python
@dataclass
class EmbedConfig:
    embedder: str | None = None
    model_id: str | None = None
    input_file: str = "dna-sequences.fasta"
    input_format: str | None = None
    output_file: str = "embedding.npy"
    region: str | None = None
    inference_batch_size: int = 16
    device: str | None = None
    csv_separator: str = "\t"
    sequence_column: str = "Seq"
```

**Class methods:**

| Method | Signature | Description |
|---|---|---|
| `from_yaml` | `(path: str \| Path) → EmbedConfig` | Load config from a YAML file. Unknown keys are silently ignored. |

**Instance methods:**

| Method | Signature | Description |
|---|---|---|
| `override_with_args` | `(ns: argparse.Namespace) → EmbedConfig` | Returns a **new** config with any non-`None` CLI values applied on top. The original instance is not mutated. |

---

## `fasta_embed.pipeline`

Orchestration layer: iterates sequences, embeds in batches, saves to disk.

### Functions

#### `run(embedder: Embedder, config: EmbedConfig) → None`

Executes a full embed-and-save pipeline:

1. Calls `iter_sequences()` to build a `SequenceBatches` iterable.
2. Logs sequence and batch counts.
3. Delegates to `vectorize()`.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `embedder` | `Embedder` | A loaded embedder instance. |
| `config` | `EmbedConfig` | Complete run configuration. |

#### `vectorize(embedder: Embedder, batches: SequenceBatches, output_file: str) → None`

Consumes batches, embeds each one, and streams results to disk.

1. Creates an `EmbeddingWriter`.
2. Iterates over batches with a `tqdm` progress bar.
3. Calls `embedder.embed_batch(batch)` for each batch.
4. Appends resulting `np.ndarray` to the writer.
5. Calls `writer.finalize()` to produce the `.npy` output.

**Parameters:**

| Name | Type | Description |
|---|---|---|
| `embedder` | `Embedder` | A loaded embedder instance. |
| `batches` | `SequenceBatches` | Iterable yielding `list[str]` batches. |
| `output_file` | `str` | Path for the `.npy` output file. |

---

## `fasta_embed.io`

Sequence I/O (FASTA / CSV loading) and embedding persistence.

### Constants

| Name | Value | Description |
|---|---|---|
| `_FASTA_EXTENSIONS` | `{".fasta", ".fa", ".fna", ".fas"}` | Recognized FASTA file extensions. |
| `_CSV_EXTENSIONS` | `{".csv", ".tsv", ".txt"}` | Recognized CSV/TSV file extensions. |

### Functions

#### `_infer_format(path: Path) → str`

Guesses file format from the extension, stripping `.gz` first.

- Returns `"fasta"` or `"csv"`.
- Raises `ValueError` if the extension is unrecognized.

#### `_iter_fasta(path: Path) → Iterator[str]`

Yields one sequence string at a time from a (possibly gzipped) FASTA file using BioPython's `SeqIO.parse`.

#### `_iter_csv(path: Path, sep: str, column: str) → Iterator[str]`

Yields one sequence string at a time from a CSV/TSV file. Uses `pandas.read_csv` with `chunksize=10_000` to limit memory.

#### `_count_fasta(path: Path) → int`

Counts FASTA records by counting lines starting with `>`. Supports gzipped files.

#### `_count_csv(path: Path) → int`

Counts data rows (total lines minus header). Supports gzipped files.

#### `iter_sequences(...) → SequenceBatches`

Main entry point for the pipeline. Returns a `SequenceBatches` iterable.

```python
def iter_sequences(
    path: str | Path,
    fmt: str | None = None,
    *,
    batch_size: int = 16,
    csv_separator: str = "\t",
    sequence_column: str = "Seq",
    region: str | None = None,
) -> SequenceBatches
```

**Parameters:**

| Name | Type | Default | Description |
|---|---|---|---|
| `path` | `str \| Path` | — | Input file path. |
| `fmt` | `str \| None` | `None` | `"fasta"` or `"csv"`. Inferred from extension when `None`. |
| `batch_size` | `int` | `16` | Max sequences per yielded batch. |
| `csv_separator` | `str` | `"\t"` | Delimiter for CSV files. |
| `sequence_column` | `str` | `"Seq"` | Column name containing sequences (CSV only). |
| `region` | `str \| None` | `None` | 16S region to extract (e.g. `"V3V4"`). |

#### `load_sequences(...) → pd.DataFrame`

Eager loader for programmatic / notebook use. Returns a DataFrame with at least a sequence column.

```python
def load_sequences(
    path: str | Path,
    fmt: str | None = None,
    *,
    csv_separator: str = "\t",
    sequence_column: str = "Seq",
) -> pd.DataFrame
```

For FASTA input, the DataFrame has columns `ID`, `SeqLen`, `Seq`.

#### `_parse_fasta(path: Path) → pd.DataFrame`

Internal helper. Parses a (possibly gzipped) FASTA file into a DataFrame with columns `ID`, `SeqLen`, `Seq`.

### Classes

#### `SequenceBatches`

Iterable that yields batches of sequences and exposes `__len__` for `tqdm` progress bars.

```python
class SequenceBatches:
    total_sequences: int  # total number of sequences in the file

    def __len__(self) -> int: ...       # number of batches (ceil division)
    def __iter__(self) -> Iterator[list[str]]: ...  # yields list[str] batches
```

**Constructor parameters:**

| Name | Type | Description |
|---|---|---|
| `path` | `Path` | Input file path. |
| `fmt` | `str` | `"fasta"` or `"csv"`. |
| `batch_size` | `int` | Max sequences per batch. |
| `csv_separator` | `str` | CSV delimiter. |
| `sequence_column` | `str` | Sequence column name. |
| `region` | `str \| None` | 16S region to extract (applied lazily during iteration). |
| `total_sequences` | `int` | Pre-counted total for progress bar. |

#### `EmbeddingWriter`

Streaming writer that accumulates embeddings via a temp binary file, then finalizes to `.npy`.

```python
class EmbeddingWriter:
    count: int  # property — total rows written so far

    def __init__(self, output_path: str | Path) -> None: ...
    def append(self, vectors: np.ndarray) -> int: ...
    def finalize(self) -> None: ...
```

**Methods:**

| Method | Returns | Description |
|---|---|---|
| `append(vectors)` | `int` | Appends a batch of embeddings (2-D `np.ndarray`) to the backing `.bin` file. Returns cumulative row count. |
| `finalize()` | `None` | Writes the `.npy` header, streams the raw binary data in 64 MB chunks, then deletes the temp file. |

---

## `fasta_embed.bio`

16S rRNA primer definitions and variable region extraction.

### Constants

#### `PRIMERS: dict[str, str]`

Maps primer names to their IUPAC-ambiguous sequences. Contains 19 primers from `27F` through `1492R`.

#### `PRIMERS_REGEX: dict[str, str]`

Maps primer names to expanded regex patterns where IUPAC ambiguity codes are replaced with character classes (e.g. `Y` → `[CT]`).

#### `_COMPILED_REGEX: dict[str, re.Pattern]`

Pre-compiled `re.Pattern` objects for each primer regex, avoiding recompilation on every call.

#### `_REGION_PRIMERS: dict[str, tuple[str, str]]`

Maps region names to `(forward_primer, reverse_primer)` pairs.

| Region | Forward | Reverse |
|---|---|---|
| V1V2 | 27F | 338R |
| V1V3 | 27F | 534R |
| V3V4 | 341F | 785R |
| V4 | 515F | 806R |
| V4V5 | 515F | 944R |
| V6V8 | 939F | 1378R |
| V7V9 | 1115F | 1492R |
| V1V8 | 27F | 1378R |
| V1V9 | 27F | 1492R |

#### `_FALLBACK_SEQUENCE = "ACGT"`

Returned when primers cannot be found in a sequence.

### Functions

#### `_find_primers(forward_name: str, reverse_name: str, seq: str) → tuple[str | None, str | None]`

Locates forward and reverse primer sequences within `seq`. The reverse primer is searched in the reverse complement of the input sequence. Returns a tuple of matched primer strings (or `None` if not found).

#### `get_region(region: str, seq: str) → str`

Extracts a 16S rRNA variable region from a full-length sequence.

**Algorithm:**
1. Look up forward/reverse primer names for the given region.
2. Find primer sites via `_find_primers()`.
3. Split the sequence after the forward primer.
4. Reverse complement, split after the reverse primer.
5. Reverse complement the result to get the extracted region.
6. Returns `_FALLBACK_SEQUENCE` ("ACGT") if any step fails.

**Raises:** `ValueError` if `region` is not a recognized region name.

---

## `fasta_embed.embedders`

### Registry Functions (from `__init__.py`)

#### `register(name: str) → Callable`

Class decorator that registers an `Embedder` subclass under `name`. Raises `ValueError` on duplicate names.

#### `create_embedder(name: str, **kwargs) → Embedder`

Looks up `name` in the registry and returns a new instance, forwarding `**kwargs` to the constructor. Raises `ValueError` if not found.

#### `list_embedders() → list[str]`

Returns a sorted list of all registered embedder names.

### Abstract Base Class (from `base.py`)

#### `Embedder` (ABC)

| Method | Abstract? | Signature | Description |
|---|---|---|---|
| `load` | Yes | `() → None` | Load model weights / initialize resources. Called once. |
| `embed` | Yes | `(sequence: str) → np.ndarray` | Return a 1-D embedding vector for a single sequence. |
| `embed_batch` | No | `(sequences: list[str]) → np.ndarray` | Return a 2-D array `(N, D)`. Default loops over `embed()`; override for batched GPU inference. |

### Concrete Implementations

#### `DNABERTEmbedder` — registered as `"dnabert"`

| Constructor Param | Type | Default |
|---|---|---|
| `model_id` | `str \| None` | `"zhihan1996/DNABERT-S"` |
| `device` | `str \| None` | Auto-detected |

#### `NTv2Embedder` — registered as `"ntv2"`

| Constructor Param | Type | Default |
|---|---|---|
| `model_id` | `str \| None` | `"InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"` |
| `device` | `str \| None` | Auto-detected |

#### `NTv3Embedder` — registered as `"ntv3"`

| Constructor Param | Type | Default |
|---|---|---|
| `model_id` | `str \| None` | `"InstaDeepAI/NTv3_100M_pre"` |
| `device` | `str \| None` | Auto-detected |

All three implementations share the same constructor signature `(model_id, device)` and override both `embed()` and `embed_batch()` for efficient GPU batching. See [Embedder System](embedders.md) for pooling strategy details.
