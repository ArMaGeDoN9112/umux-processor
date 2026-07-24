"""Command-line interface foundation for the UMUX-Lite processor."""

from __future__ import annotations

import argparse
from pathlib import Path

from umux_processor.config import ConfigurationError, load_configuration
from umux_processor.logging import configure_logging


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without performing application side effects."""
    parser = argparse.ArgumentParser(description="UMUX-Lite processing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config") / "normalization.toml",
        help="Path to the TOML normalization configuration (default: %(default)s)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate configuration for the placeholder command-line entry point."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        load_configuration(args.config)
    except ConfigurationError as error:
        build_parser().error(str(error))
    return 0
