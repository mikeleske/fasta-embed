# Embedder System

## Overview

The embedder system provides a pluggable architecture for DNA sequence embedding models. It consists of three parts:

1. **Abstract interface** (`base.py`) — defines the contract all backends must fulfill.
2. **Registry** (`__init__.py`) — maps string names to concrete classes at runtime.
3. **Concrete backends** — each file implements one model family.

---

## Class Hierarchy

```mermaid
classDiagram
    class Embedder {
        <<abstract>>
        +load()* void
        +embed(sequence: str)* np.ndarray
        +embed_batch(sequences: list~str~) np.ndarray
    }

    class DNABERTEmbedder {
        -DEFAULT_MODEL_ID = "zhihan1996/DNABERT-S"
        -model_id: str
        -device: torch.device
        -model: AutoModel
        -tokenizer: AutoTokenizer
        +load() void
        +embed(sequence: str) np.ndarray
        +embed_batch(sequences: list~str~) np.ndarray
    }

    class NTv2Embedder {
        -DEFAULT_MODEL_ID = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
        -model_id: str
        -device: torch.device
        -model: AutoModelForMaskedLM
        -tokenizer: AutoTokenizer
        +load() void
        +embed(sequence: str) np.ndarray
        +embed_batch(sequences: list~str~) np.ndarray
    }

    class NTv3Embedder {
        -DEFAULT_MODEL_ID = "InstaDeepAI/NTv3_100M_pre"
        -model_id: str
        -device: torch.device
        -model: AutoModelForMaskedLM
        -tokenizer: AutoTokenizer
        +load() void
        +embed(sequence: str) np.ndarray
        +embed_batch(sequences: list~str~) np.ndarray
    }

    class Evo2Embedder {
        -DEFAULT_MODEL_ID = "evo2_1b_base"
        -DEFAULT_LAYER_NAME = "blocks.20.mlp.l3"
        -model_id: str
        -layer_name: str
        -device: torch.device
        -_model: Evo2
        +load() void
        +embed(sequence: str) np.ndarray
        +embed_batch(sequences: list~str~) np.ndarray
    }

    Embedder <|-- DNABERTEmbedder : "@register('dnabert')"
    Embedder <|-- NTv2Embedder : "@register('ntv2')"
    Embedder <|-- NTv3Embedder : "@register('ntv3')"
    Embedder <|-- Evo2Embedder : "@register('evo2')"
```

---

## Registry Pattern

The registry is a module-level dictionary `_REGISTRY: dict[str, type[Embedder]]` that maps names to classes. Registration happens at import time via a `@register` decorator.

```mermaid
flowchart LR
    subgraph "Import Time"
        A["embedders/__init__.py<br/>imports dnabert, ntv2, ntv3, evo2"] --> B["@register('dnabert')<br/>executes on DNABERTEmbedder"]
        A --> C["@register('ntv2')<br/>executes on NTv2Embedder"]
        A --> D["@register('ntv3')<br/>executes on NTv3Embedder"]
        A --> E2["@register('evo2')<br/>executes on Evo2Embedder"]
    end

    subgraph "_REGISTRY"
        R["{ 'dnabert': DNABERTEmbedder,<br/>'evo2': Evo2Embedder,<br/>'ntv2': NTv2Embedder,<br/>'ntv3': NTv3Embedder }"]
    end

    B --> R
    C --> R
    D --> R
    E2 --> R

    subgraph "Runtime"
        E["create_embedder('ntv3',<br/>device='cuda:0')"] --> F["_REGISTRY['ntv3']<br/>(device='cuda:0')"]
        F --> G["NTv3Embedder instance"]
    end

    R --> E
```

### Registry API

| Function | Signature | Description |
|---|---|---|
| `register` | `register(name: str) → Callable` | Class decorator that adds an Embedder subclass to the registry. |
| `create_embedder` | `create_embedder(name: str, **kwargs) → Embedder` | Instantiates a registered embedder, forwarding kwargs to its constructor. |
| `list_embedders` | `list_embedders() → list[str]` | Returns sorted list of registered embedder names. |

---

## Built-in Backends

### DNABERT-S (`dnabert`)

| Property | Value |
|---|---|
| Default model | `zhihan1996/DNABERT-S` |
| Model class | `AutoModel` |
| Pooling | Mean pooling over first hidden state, masked by attention |
| Tokenization | Standard `AutoTokenizer` with padding |

```mermaid
flowchart LR
    SEQ["DNA Sequence"] --> TOK["AutoTokenizer<br/>(padding=True)"]
    TOK --> IDS["input_ids + attention_mask"]
    IDS --> MODEL["AutoModel<br/>DNABERT-S"]
    MODEL --> HS["hidden_states[0]<br/>(first output)"]
    HS --> POOL["Masked mean pooling<br/>(hs × mask).sum / mask.sum"]
    POOL --> VEC["Embedding vector<br/>(1-D np.ndarray)"]
```

### Nucleotide Transformer v2 (`ntv2`)

| Property | Value |
|---|---|
| Default model | `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species` |
| Model class | `AutoModelForMaskedLM` |
| Pooling | Attention-mask-weighted mean pooling over last hidden state |
| Loading | Uses `device_map` for placement |

```mermaid
flowchart LR
    SEQ["DNA Sequence"] --> TOK["AutoTokenizer<br/>(padding=True)"]
    TOK --> IDS["input_ids"]
    IDS --> MASK["Recompute attention_mask<br/>from pad_token_id"]
    IDS --> MODEL["AutoModelForMaskedLM<br/>NT v2"]
    MASK --> MODEL
    MODEL --> HS["hidden_states[-1]<br/>(last hidden state)"]
    HS --> POOL["Masked mean pooling"]
    POOL --> VEC["Embedding vector"]
```

### Nucleotide Transformer v3 (`ntv3`)

| Property | Value |
|---|---|
| Default model | `InstaDeepAI/NTv3_100M_pre` |
| Model class | `AutoModelForMaskedLM` |
| Pooling | Same as NT v2 |
| Special | `add_special_tokens=False`, `pad_to_multiple_of=128` |
| Loading | Uses `device_map` for placement |

```mermaid
flowchart LR
    SEQ["DNA Sequence"] --> TOK["AutoTokenizer<br/>(no special tokens,<br/>pad_to_multiple_of=128)"]
    TOK --> IDS["input_ids"]
    IDS --> MASK["Recompute attention_mask<br/>from pad_token_id"]
    IDS --> MODEL["AutoModelForMaskedLM<br/>NT v3"]
    MASK --> MODEL
    MODEL --> HS["hidden_states[-1]<br/>(last hidden state)"]
    HS --> POOL["Masked mean pooling"]
    POOL --> VEC["Embedding vector"]
```

### Evo 2 (`evo2`)

| Property | Value |
|---|---|
| Default checkpoint | `evo2_1b_base` |
| Architecture | StripedHyena 2 (autoregressive, character-level) |
| Default layer | `blocks.20.mlp.l3` (~80 % network depth) |
| Pooling | Unmasked mean pooling over all sequence positions |
| Batching | Sequential (one sequence per forward pass) |
| GPU requirement | FP8 + Transformer Engine (Hopper GPU) for `1b_base` / `20b` / `40b`; bfloat16 for `7b` variants |
| Install | `pip install flash-attn==2.8.0.post2 --no-build-isolation && pip install evo2` |

```mermaid
flowchart LR
    SEQ["DNA Sequence"] --> TOK["Evo2 character-level<br/>tokenizer.tokenize(seq)"]
    TOK --> IDS["input_ids tensor<br/>(1, seq_len) → GPU"]
    IDS --> MODEL["Evo2 model<br/>(StripedHyena 2)<br/>return_embeddings=True"]
    MODEL --> HS["layer_embeddings[layer_name]<br/>shape: (1, seq_len, D)"]
    HS --> POOL["Mean pool over<br/>sequence positions<br/>squeeze(0).mean(dim=0)"]
    POOL --> FLOAT["Cast to float32"]
    FLOAT --> VEC["Embedding vector<br/>(1-D np.ndarray)"]
```

> **Note:** Evo 2 is an autoregressive generative model (not masked-LM), so there is no padding mask. All token positions contribute equally to the mean pool. The `evo2` package must be installed separately — see the module docstring or [github.com/arcinstitute/evo2](https://github.com/arcinstitute/evo2) for full instructions.

#### Choosing a Layer

The Evo 2 paper demonstrates that intermediate layers produce better task-specific embeddings than the final layer. Layer names follow the pattern `blocks.{N}.mlp.l3`. A useful heuristic:

| Use case | Suggested depth |
|---|---|
| Sequence composition / motif features | ~50 % depth (earlier blocks) |
| Functional / taxonomic embeddings | ~80 % depth (default) |
| Generative likelihood features | ~95 % depth (near output) |

Override the default via the `layer_name` constructor parameter:

```python
from fasta_embed.embedders.evo2 import Evo2Embedder

embedder = Evo2Embedder(
    model_id="evo2_7b",
    device="cuda:0",
    layer_name="blocks.28.mlp.l3",  # ~90% depth for the 7B model
)
embedder.load()
```

#### Available Checkpoints

| Checkpoint | Parameters | Context | FP8 Required |
|---|---|---|---|
| `evo2_1b_base` | 1B | 8K | Yes (Hopper GPU) |
| `evo2_7b_base` | 7B | 8K | No |
| `evo2_7b` | 7B | 1M | No |
| `evo2_7b_262k` | 7B | 262K | No |
| `evo2_20b` | 20B | 1M | Yes |
| `evo2_40b` | 40B | 1M | Yes |

---

## Comparison Table

| Feature | DNABERT-S | NT v2 | NT v3 | Evo 2 |
|---|---|---|---|---|
| Architecture | BERT (encoder) | BERT (masked-LM) | BERT (masked-LM) | StripedHyena 2 (autoregressive) |
| Hidden state used | First (`[0]`) | Last (`[-1]`) | Last (`[-1]`) | Named intermediate layer |
| Model API | `AutoModel` | `AutoModelForMaskedLM` | `AutoModelForMaskedLM` | `Evo2` (custom) |
| Special tokens | Default | Default | Disabled | N/A (char-level) |
| Padding strategy | Standard | Standard | Multiple of 128 | None (sequential) |
| Attention mask source | Tokenizer output | Recomputed from `pad_token_id` | Recomputed from `pad_token_id` | Not used |
| Batch inference | True batched | True batched | True batched | Sequential |
| Install source | HuggingFace | HuggingFace | HuggingFace | `pip install evo2` |
