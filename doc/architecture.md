# Architecture Overview

## System Context

fasta-embed is a command-line tool and Python library that converts DNA sequences into dense vector embeddings using transformer-based language models. It reads sequences from FASTA or CSV files, optionally extracts 16S rRNA variable regions, generates embeddings in batches via a pluggable model backend, and streams the output to a NumPy `.npy` file.

```mermaid
C4Context
    title System Context — fasta-embed

    Person(user, "Researcher", "Runs embedding jobs via CLI or Python API")

    System(fastaembed, "fasta-embed", "Reads DNA sequences, generates embeddings via transformer models, writes .npy output")

    System_Ext(hf, "HuggingFace Hub", "Hosts pretrained model weights and tokenizers")
    SystemDb_Ext(input, "Input Files", "FASTA / CSV / TSV (optionally gzipped)")
    SystemDb_Ext(output, "Output File", "NumPy .npy embedding matrix")

    Rel(user, fastaembed, "Configures & runs")
    Rel(fastaembed, hf, "Downloads model weights on first use")
    Rel(fastaembed, input, "Reads sequences from")
    Rel(fastaembed, output, "Writes embeddings to")
```

---

## Module Map

The package is organized into five functional layers. Each layer has a single responsibility and communicates with adjacent layers through well-defined interfaces.

```mermaid
graph TB
    subgraph CLI ["CLI Layer"]
        main["__main__.py<br/><i>Argument parsing, logging setup,<br/>embedder creation, pipeline launch</i>"]
    end

    subgraph Config ["Configuration Layer"]
        config["config.py<br/><i>EmbedConfig dataclass<br/>YAML + CLI override merging</i>"]
    end

    subgraph Orchestration ["Orchestration Layer"]
        pipeline["pipeline.py<br/><i>run() and vectorize()<br/>Coordinates I/O with embedding</i>"]
    end

    subgraph IO ["I/O Layer"]
        io["io.py<br/><i>Format detection, FASTA/CSV iteration,<br/>SequenceBatches, EmbeddingWriter</i>"]
        bio["bio.py<br/><i>16S primer definitions,<br/>variable region extraction</i>"]
    end

    subgraph Embedders ["Embedding Layer"]
        registry["embedders/__init__.py<br/><i>Registry, factory, auto-import</i>"]
        base["embedders/base.py<br/><i>Abstract Embedder ABC</i>"]
        dnabert["embedders/dnabert.py<br/><i>DNABERT-S backend</i>"]
        ntv2["embedders/ntv2.py<br/><i>NT v2 backend</i>"]
        ntv3["embedders/ntv3.py<br/><i>NT v3 backend</i>"]
    end

    main --> config
    main --> registry
    main --> pipeline

    pipeline --> config
    pipeline --> io
    pipeline --> base

    io --> bio

    registry --> base
    dnabert --> base
    ntv2 --> base
    ntv3 --> base
    dnabert -.->|"@register('dnabert')"| registry
    ntv2 -.->|"@register('ntv2')"| registry
    ntv3 -.->|"@register('ntv3')"| registry

    style CLI fill:#e3f2fd,stroke:#1565c0
    style Config fill:#fff3e0,stroke:#e65100
    style Orchestration fill:#e8f5e9,stroke:#2e7d32
    style IO fill:#fce4ec,stroke:#c62828
    style Embedders fill:#f3e5f5,stroke:#6a1b9a
```

---

## Component Interaction Sequence

This diagram shows the runtime interaction between components during a typical embedding run.

```mermaid
sequenceDiagram
    actor User
    participant CLI as __main__
    participant Cfg as EmbedConfig
    participant Reg as embedders registry
    participant Emb as Embedder (concrete)
    participant Pipe as pipeline
    participant IO as io.py
    participant Bio as bio.py
    participant Writer as EmbeddingWriter

    User->>CLI: python -m fasta_embed --config config.yaml
    CLI->>Cfg: from_yaml("config.yaml")
    Cfg-->>CLI: EmbedConfig instance
    CLI->>Cfg: override_with_args(cli_args)
    Cfg-->>CLI: merged EmbedConfig

    CLI->>Reg: create_embedder(name, device, model_id)
    Reg-->>CLI: Embedder instance
    CLI->>Emb: load()
    Note over Emb: Downloads / loads model weights<br/>and tokenizer from HuggingFace

    CLI->>Pipe: run(embedder, config)
    Pipe->>IO: iter_sequences(path, fmt, batch_size, region, ...)
    IO->>IO: _infer_format(path)
    IO->>IO: _count_fasta(path) or _count_csv(path)
    IO-->>Pipe: SequenceBatches

    Pipe->>Writer: EmbeddingWriter(output_path)

    loop For each batch
        Pipe->>IO: next(SequenceBatches)
        IO->>IO: _iter_fasta() or _iter_csv()
        opt region extraction enabled
            IO->>Bio: get_region(region, seq)
            Bio-->>IO: extracted subsequence
        end
        IO-->>Pipe: list[str] (batch of sequences)
        Pipe->>Emb: embed_batch(sequences)
        Emb-->>Pipe: np.ndarray (N × D)
        Pipe->>Writer: append(embeddings)
    end

    Pipe->>Writer: finalize()
    Note over Writer: Converts raw binary → .npy<br/>in 64 MB streaming chunks
    Writer-->>Pipe: ✓ embeddings.npy written
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Registry pattern for embedders** | New backends are added by decorating a class with `@register("name")` — no factory switch statements to maintain. |
| **Streaming binary writer** | `EmbeddingWriter` appends raw bytes to a temp `.bin` file, then streams the final `.npy` header + data in 64 MB chunks. Peak memory stays constant regardless of dataset size. |
| **Two-pass sequence reading** | A fast counting pass (`_count_fasta` / `_count_csv`) runs first so `tqdm` gets an accurate total; the second pass iterates lazily. |
| **Config layering** | YAML provides reproducible defaults; CLI flags override individual values. The `override_with_args` method returns a new immutable config. |
| **Region extraction in the iterator** | 16S region extraction (`bio.get_region`) is applied lazily inside `SequenceBatches.__iter__`, avoiding a separate preprocessing step. |
