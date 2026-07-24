"""Command-line interface foundation for the UMUX-Lite processor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from umux_processor.artifacts import ArtifactWriteError
from umux_processor.config import ConfigurationError, load_configuration
from umux_processor.ingestion import IngestionError
from umux_processor.logging import configure_logging
from umux_processor.pipeline import PipelineServiceError, run_pipeline_service


EXIT_SUCCESS = 0
EXIT_UNEXPECTED_FAILURE = 1
EXIT_INVALID_ARGUMENTS = 2
EXIT_DATA_OR_CONFIGURATION_ERROR = 3
EXIT_OUTPUT_ERROR = 4


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser without performing application side effects."""
    parser = argparse.ArgumentParser(description="UMUX-Lite processing pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config") / "normalization.toml",
        help="Path to the TOML normalization configuration (default: %(default)s)",
    )
    parser.add_argument(
        "--input", nargs="+", required=True, metavar="PATH_OR_GLOB",
        help="One or more CSV paths or glob patterns",
    )
    parser.add_argument(
        "--output", type=Path, required=True, metavar="DIRECTORY",
        help="Directory where audit artifacts and dashboard are written",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the pipeline CLI and return its process exit status."""
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)
    try:
        config = load_configuration(args.config)
        execution = run_pipeline_service(args.input, args.output, config)
    except ConfigurationError as error:
        logging.getLogger(__name__).error("Configuration error: %s", error)
        return EXIT_DATA_OR_CONFIGURATION_ERROR
    except IngestionError as error:
        logging.getLogger(__name__).error("Input error: %s", error)
        return EXIT_DATA_OR_CONFIGURATION_ERROR
    except (PipelineServiceError, ArtifactWriteError) as error:
        logging.getLogger(__name__).error("Output error: %s", error)
        return EXIT_OUTPUT_ERROR
    except Exception:
        logging.getLogger(__name__).exception("Pipeline failed unexpectedly")
        return EXIT_UNEXPECTED_FAILURE

    overall = execution.result.quality.overall.iloc[0]
    duplicate_count = int(
        execution.result.quality.by_rejection_reason.loc[
            execution.result.quality.by_rejection_reason["rejection_reason"].isin(
                ["duplicate_exact", "duplicate_conflict"]
            ),
            "rejected_row_count",
        ].sum()
    )
    print(
        "Pipeline completed: "
        f"raw={int(overall['raw_row_count'])} "
        f"accepted={int(overall['accepted_row_count'])} "
        f"rejected={int(overall['rejected_row_count'])} "
        f"duplicates={duplicate_count} "
        f"output={args.output}"
    )
    return EXIT_SUCCESS
