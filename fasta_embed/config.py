"""Dataclass-based configuration with YAML and CLI override support."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass
class EmbedConfig:
    """Central configuration for a fasta-embed run.

    Instances can be built from YAML files, CLI arguments, or a combination
    of both (YAML provides defaults, CLI overrides individual values).
    """

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

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> EmbedConfig:
        """Load configuration from a YAML file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def override_with_args(self, ns: argparse.Namespace) -> EmbedConfig:
        """Return a *new* config with any non-``None`` CLI values applied on
        top of the current instance."""
        updates: dict = {}
        for f in fields(self):
            cli_val = getattr(ns, f.name, None)
            if cli_val is not None:
                updates[f.name] = cli_val
            else:
                updates[f.name] = getattr(self, f.name)
        return EmbedConfig(**updates)
