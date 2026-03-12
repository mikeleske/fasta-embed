# Configuration

## Overview

fasta-embed uses a layered configuration system built on a Python `dataclass`. Configuration can come from three sources, applied in order of increasing precedence:

1. **Hardcoded defaults** — defined in `EmbedConfig` field defaults.
2. **YAML config file** — loaded via `--config config.yaml`.
3. **CLI flags** — any non-`None` value overrides both defaults and YAML.

---

## Precedence Diagram

```mermaid
flowchart TD
    A["Hardcoded Defaults<br/><code>EmbedConfig()</code><br/><i>Lowest precedence</i>"] --> B["YAML File<br/><code>EmbedConfig.from_yaml(path)</code><br/><i>Overrides defaults</i>"]
    B --> C["CLI Flags<br/><code>config.override_with_args(ns)</code><br/><i>Highest precedence</i>"]
    C --> D["Final EmbedConfig<br/><i>Used by pipeline</i>"]

    style A fill:#e3f2fd,stroke:#1565c0
    style B fill:#fff3e0,stroke:#e65100
    style C fill:#e8f5e9,stroke:#2e7d32
    style D fill:#f3e5f5,stroke:#6a1b9a
```

### How Merging Works

```mermaid
flowchart LR
    subgraph "override_with_args(namespace)"
        direction TB
        F["For each field in EmbedConfig"] --> G{"CLI value<br/>is not None?"}
        G -->|Yes| H["Use CLI value"]
        G -->|No| I["Keep current value<br/>(YAML or default)"]
        H --> J["Build new EmbedConfig<br/>from merged dict"]
        I --> J
    end
```

The method returns a **new** `EmbedConfig` instance — the original is never mutated.

---

## Configuration Reference

| Key | CLI Flag | Type | Default | Description |
|---|---|---|---|---|
| `embedder` | `--embedder` | `str \| None` | `None` | Embedder backend name (`dnabert`, `ntv2`, `ntv3`). |
| `model_id` | `--model-id` | `str \| None` | `None` | HuggingFace model ID or local path. When `None`, each embedder uses its own default. |
| `input_file` | `--input` | `str` | `dna-sequences.fasta` | Path to the input FASTA or CSV/TSV file. |
| `input_format` | `--input-format` | `str \| None` | `None` | `"fasta"` or `"csv"`. Auto-detected from extension when `None`. |
| `output_file` | `--output` | `str` | `embedding.npy` | Path for the output NumPy `.npy` file. |
| `region` | `--region` | `str \| None` | `None` | 16S variable region to extract before embedding. `None` = full sequence. |
| `inference_batch_size` | `--inference-batch-size` | `int` | `16` | Number of sequences per model forward pass. |
| `device` | `--device` | `str \| None` | `None` | PyTorch device string (e.g. `cuda:0`, `cpu`). Auto-detected when `None`. |
| `csv_separator` | `--csv-separator` | `str` | `"\t"` | Column delimiter for CSV input. |
| `sequence_column` | `--sequence-column` | `str` | `"Seq"` | Column name containing DNA sequences (CSV only). |

---

## YAML Config File

The example configuration file (`config.example.yaml`) documents every available key:

```yaml
embedder: ntv3
model_id: null
input_file: dna-sequences.fasta
input_format: null
output_file: embedding.npy
region: null
inference_batch_size: 16
device: null
csv_separator: "\t"
sequence_column: Seq
```

Only keys matching `EmbedConfig` field names are loaded — unknown keys are silently ignored.

---

## Supported 16S Regions

When `region` is set, `bio.get_region()` extracts the corresponding variable region using PCR primer positions. If primers cannot be located, a fallback sequence `"ACGT"` is returned.

```mermaid
flowchart LR
    subgraph "16S rRNA Gene (~1500 bp)"
        direction LR
        V1["V1"] --- V2["V2"] --- V3["V3"] --- V4["V4"] --- V5["V5"] --- V6["V6"] --- V7["V7"] --- V8["V8"] --- V9["V9"]
    end

    R1["V1V2<br/>27F → 338R"] -.-> V1
    R1 -.-> V2
    R2["V1V3<br/>27F → 534R"] -.-> V1
    R2 -.-> V3
    R3["V3V4<br/>341F → 785R"] -.-> V3
    R3 -.-> V4
    R4["V4<br/>515F → 806R"] -.-> V4
    R5["V4V5<br/>515F → 944R"] -.-> V4
    R5 -.-> V5
    R6["V6V8<br/>939F → 1378R"] -.-> V6
    R6 -.-> V8
    R7["V7V9<br/>1115F → 1492R"] -.-> V7
    R7 -.-> V9
    R8["V1V8<br/>27F → 1378R"] -.-> V1
    R8 -.-> V8
    R9["V1V9<br/>27F → 1492R"] -.-> V1
    R9 -.-> V9
```

| Region | Forward Primer | Reverse Primer |
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

---

## Device Auto-Detection

When `device` is `None`, each embedder resolves it at `load()` time:

```mermaid
flowchart TD
    A{"device parameter<br/>provided?"} -->|Yes| B["Use specified device"]
    A -->|No| C{"torch.cuda.is_available()?"}
    C -->|Yes| D["Use 'cuda'"]
    C -->|No| E["Use 'cpu'"]
```
