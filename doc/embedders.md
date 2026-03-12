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

    Embedder <|-- DNABERTEmbedder : "@register('dnabert')"
    Embedder <|-- NTv2Embedder : "@register('ntv2')"
    Embedder <|-- NTv3Embedder : "@register('ntv3')"
```

---

## Registry Pattern

The registry is a module-level dictionary `_REGISTRY: dict[str, type[Embedder]]` that maps names to classes. Registration happens at import time via a `@register` decorator.

```mermaid
flowchart LR
    subgraph "Import Time"
        A["embedders/__init__.py<br/>imports dnabert, ntv2, ntv3"] --> B["@register('dnabert')<br/>executes on DNABERTEmbedder"]
        A --> C["@register('ntv2')<br/>executes on NTv2Embedder"]
        A --> D["@register('ntv3')<br/>executes on NTv3Embedder"]
    end

    subgraph "_REGISTRY"
        R["{ 'dnabert': DNABERTEmbedder,<br/>'ntv2': NTv2Embedder,<br/>'ntv3': NTv3Embedder }"]
    end

    B --> R
    C --> R
    D --> R

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

---

## Comparison Table

| Feature | DNABERT-S | NT v2 | NT v3 |
|---|---|---|---|
| Hidden state used | First (`[0]`) | Last (`[-1]`) | Last (`[-1]`) |
| Model API | `AutoModel` | `AutoModelForMaskedLM` | `AutoModelForMaskedLM` |
| Special tokens | Default | Default | Disabled |
| Padding strategy | Standard | Standard | Multiple of 128 |
| Attention mask source | Tokenizer output | Recomputed from `pad_token_id` | Recomputed from `pad_token_id` |
| `device_map` | Manual `.to(device)` | `{"": device}` | `{"": device}` |
| `output_hidden_states` | Not needed | `True` | `True` |
