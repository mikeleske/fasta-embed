# fasta-embed Documentation

**Embed DNA sequences into vector space using pluggable transformer backends.**

---

## Table of Contents

| Document | Description |
|---|---|
| [Architecture Overview](architecture.md) | High-level system design, module map, and component relationships |
| [Pipeline & Data Flow](pipeline.md) | End-to-end execution flow from CLI invocation to `.npy` output |
| [Embedder System](embedders.md) | Registry pattern, abstract interface, and built-in model backends |
| [Configuration](configuration.md) | YAML + CLI configuration system with layered precedence |
| [API Reference](api-reference.md) | Module-level API documentation for every public function and class |
| [Extending fasta-embed](extending.md) | Step-by-step guide for adding custom embedder backends |

---

## Project at a Glance

```
fasta-embed/
├── config.example.yaml          # Reference configuration file
├── requirements.txt             # Python dependencies
└── fasta_embed/                 # Main package
    ├── __main__.py              # CLI entry point
    ├── config.py                # Dataclass-based configuration
    ├── bio.py                   # 16S rRNA primer definitions & region extraction
    ├── io.py                    # Sequence I/O and embedding persistence
    ├── pipeline.py              # Orchestration layer
    └── embedders/               # Pluggable embedding backends
        ├── __init__.py          # Registry & factory
        ├── base.py              # Abstract Embedder interface
        ├── dnabert.py           # DNABERT-S backend
        ├── ntv2.py              # Nucleotide Transformer v2 backend
        └── ntv3.py              # Nucleotide Transformer v3 backend
```

## Dependencies

| Package | Role |
|---|---|
| `numpy` | Embedding arrays, `.npy` serialization |
| `pandas` | CSV/TSV reading with chunked iteration |
| `torch` | GPU/CPU tensor computation |
| `transformers` | HuggingFace model & tokenizer loading |
| `biopython` | FASTA parsing, reverse complement |
| `accelerate` | Efficient HuggingFace model loading |
| `einops` | Tensor reshaping (used by NT models) |
| `pyyaml` | YAML configuration parsing |
| `tqdm` | Progress bar for batch processing |
