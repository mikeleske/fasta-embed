# fasta-embed

**Embed your DNA sequences into vector space**

Generate dense embeddings for DNA sequences using pluggable model backends
(DNABERT-S, Nucleotide Transformer, NTv3, or your own).

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Using CLI flags
python -m fasta_embed --embedder ntv3 --input sequences.fasta --output embeddings.npy

# Using a config file
cp config.example.yaml config.yaml   # edit as needed
python -m fasta_embed --config config.yaml

# Mix both (CLI flags override the config file)
python -m fasta_embed --config config.yaml --region V3V4 --device cuda:0
```

## Available embedders

| Name                     | Default model                                           |
|--------------------------|---------------------------------------------------------|
| `dnabert`                | `zhihan1996/DNABERT-S`                                  |
| `nucleotide-transformer` | `InstaDeepAI/nucleotide-transformer-2.5b-multi-species` |
| `ntv3`                   | `InstaDeepAI/NTv3_8M_pre`                               |

List them at any time:

```bash
python -m fasta_embed --list-embedders
```

## Configuration

All options can be set via YAML config file, CLI flags, or both (CLI wins).
See `config.example.yaml` for the full reference.

Key options:

| Flag                      | Config key               | Description                             |
|---------------------------|--------------------------|-----------------------------------------|
| `--embedder`              | `embedder`               | Embedder strategy name                  |
| `--model-id`              | `model_id`               | HuggingFace model ID or local path      |
| `--input`                 | `input_file`             | Input FASTA or CSV file                 |
| `--input-format`          | `input_format`           | `fasta` or `csv`                        |
| `--output`                | `output_file`            | Output `.npy` file                      |
| `--region`                | `region`                 | 16S region (e.g. V3V4, V4, V1V9)       |
| `--inference-batch-size`  | `inference_batch_size`   | Sequences per forward pass (default 16) |
| `--save-batch-size`       | `save_batch_size`        | Flush-to-disk threshold (default 10000) |
| `--device`                | `device`                 | Torch device (auto-detected if omitted) |

## Adding a custom embedder

1. Create a new file in `fasta_embed/embedders/`, e.g. `my_model.py`.
2. Subclass `Embedder` and decorate with `@register`:

```python
from fasta_embed.embedders import register
from fasta_embed.embedders.base import Embedder

@register("my-model")
class MyModelEmbedder(Embedder):
    def __init__(self, model_id: str = "default/id", device: str | None = None):
        ...

    def load(self) -> None:
        ...

    def embed(self, sequence: str) -> np.ndarray:
        ...
```

3. Import the module in `fasta_embed/embedders/__init__.py`:

```python
from . import my_model  # noqa
```

4. Use it: `python -m fasta_embed --embedder my-model --input seqs.fasta`
