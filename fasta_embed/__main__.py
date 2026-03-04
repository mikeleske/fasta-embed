"""CLI entry-point: ``python -m fasta_embed``."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import EmbedConfig
from .embedders import create_embedder, list_embedders
from .pipeline import run


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="fasta_embed",
        description="Embed DNA sequences into vector space.",
    )
    p.add_argument(
        "--config",
        metavar="FILE",
        help="YAML config file (CLI flags override values from this file)",
    )
    p.add_argument("--embedder", metavar="NAME", help="Embedder strategy name")
    p.add_argument(
        "--model-id",
        metavar="ID",
        dest="model_id",
        help="HuggingFace model ID or local path",
    )
    p.add_argument(
        "--input", metavar="FILE", dest="input_file", help="Input FASTA or CSV file"  # noqa: E501
    )
    p.add_argument(
        "--input-format",
        metavar="FMT",
        dest="input_format",
        choices=["fasta", "csv"],
        help="Input format: fasta or csv",
    )
    p.add_argument(
        "--output", metavar="FILE", dest="output_file", help="Output .npy file path"  # noqa: E501
    )
    p.add_argument(
        "--region", metavar="REGION", help="16S region to extract (e.g. V3V4)"
    )
    p.add_argument(
        "--inference-batch-size",
        metavar="N",
        dest="inference_batch_size",
        type=int,
        help="Sequences per model forward pass (default: 16)",
    )
    p.add_argument(
        "--save-batch-size",
        metavar="N",
        dest="save_batch_size",
        type=int,
        help="Embeddings to accumulate before saving to disk (default: 10000)",
    )
    p.add_argument("--device", metavar="DEV", help="Torch device, e.g. cuda:0 or cpu")  # noqa: E501
    p.add_argument(
        "--csv-separator",
        metavar="SEP",
        dest="csv_separator",
        help="CSV column separator (default: tab)",
    )
    p.add_argument(
        "--sequence-column",
        metavar="COL",
        dest="sequence_column",
        help="Column name containing sequences (default: Seq)",
    )
    p.add_argument(
        "--list-embedders",
        action="store_true",
        help="Print available embedders and exit",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.list_embedders:
        names = list_embedders()
        print("Available embedders:")
        for name in names:
            print(f"  - {name}")
        sys.exit(0)

    config = EmbedConfig.from_yaml(args.config) if args.config else EmbedConfig()  # noqa: E501
    config = config.override_with_args(args)

    kwargs: dict = {"device": config.device}
    if config.model_id is not None:
        kwargs["model_id"] = config.model_id
    embedder = create_embedder(config.embedder, **kwargs)
    embedder.load()

    device = getattr(embedder, "device", None)
    if device is not None:
        using_gpu = "cuda" in str(device)
        print(f"Using device: {device} ({'GPU' if using_gpu else 'CPU'})")
    else:
        print("Using device: unknown (embedder does not expose a device attribute)")  # noqa: E501

    run(embedder, config)


if __name__ == "__main__":
    main()
