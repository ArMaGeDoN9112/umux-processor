"""The in-process UMUX pipeline result used by reporting adapters.

This module deliberately has no CLI or filesystem orchestration.  It combines
the completed Tasks 1--6 transformations into one immutable result so output
adapters can consume a consistent, reconciled set of dataframes.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from umux_processor.aggregation import (
    DataQualitySummaries,
    build_data_quality_summaries,
    build_monthly_aggregates,
    build_product_summaries,
)
from umux_processor.cleaning import clean_records, deduplicate_records
from umux_processor.config import NormalizationConfig
from umux_processor.scoring import score_accepted_records


LOGGER = logging.getLogger(__name__)


class PipelineServiceError(RuntimeError):
    """Raised when orchestration cannot safely create a complete output set."""


@dataclass(frozen=True)
class PipelineResult:
    """Final row dispositions and analytical results for one input batch."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    monthly_aggregates: pd.DataFrame
    product_summary: pd.DataFrame
    quality: DataQualitySummaries


@dataclass(frozen=True)
class PipelineExecution:
    """The complete reusable input-to-artifact pipeline execution."""

    result: PipelineResult
    input_files: tuple[Path, ...]
    artifact_paths: Mapping[str, Path]


def run_pipeline(records: pd.DataFrame, config: NormalizationConfig) -> PipelineResult:
    """Run cleaning, duplicate resolution, scoring, and aggregation in order."""
    cleaned = clean_records(records, config)
    LOGGER.info("validation: raw=%d currently_valid=%d rejected=%d", cleaned.input_count, len(cleaned.currently_valid), len(cleaned.rejected))
    deduplicated = deduplicate_records(cleaned)
    LOGGER.info("deduplication: accepted=%d rejected=%d", len(deduplicated.accepted), len(deduplicated.rejected))
    accepted = score_accepted_records(deduplicated.accepted)
    LOGGER.info("scoring: scored=%d", len(accepted))
    monthly_aggregates = build_monthly_aggregates(accepted)
    product_summary = build_product_summaries(monthly_aggregates)
    quality = build_data_quality_summaries(accepted, deduplicated.rejected)
    LOGGER.info("aggregation: monthly_groups=%d product_summaries=%d", len(monthly_aggregates), len(product_summary))
    return PipelineResult(
        accepted=accepted,
        rejected=deduplicated.rejected,
        monthly_aggregates=monthly_aggregates,
        product_summary=product_summary,
        quality=quality,
    )


def run_pipeline_service(
    inputs: Sequence[str | Path], output_directory: str | Path, config: NormalizationConfig
) -> PipelineExecution:
    """Ingest inputs, run all analytic stages, and write the complete artifact set.

    This is the Python-facing orchestration API. It deliberately accepts typed
    configuration and never parses command-line arguments or exits the caller.
    Input/schema failures retain their :class:`IngestionError`; unrecoverable
    output failures are translated to :class:`PipelineServiceError`.
    """
    from umux_processor.artifacts import ARTIFACT_FILENAMES, ArtifactWriteError, write_audit_artifacts
    from umux_processor.ingestion import ingest_csv_inputs

    ingested = ingest_csv_inputs(inputs)
    LOGGER.info("ingestion: files=%d raw=%d", len(ingested.input_files), len(ingested.records))
    result = run_pipeline(ingested.records, config)
    try:
        artifact_paths = write_audit_artifacts(
            result, output_directory, small_sample_threshold=config.report.small_sample_threshold
        )
    except ArtifactWriteError as error:
        raise PipelineServiceError(str(error)) from error

    expected = {*ARTIFACT_FILENAMES, "dashboard.html"}
    if set(artifact_paths) != expected or not all(path.is_file() for path in artifact_paths.values()):
        raise PipelineServiceError(f"Could not write complete artifact set to {output_directory}")
    LOGGER.info("artifacts: output=%s files=%d", output_directory, len(artifact_paths))
    return PipelineExecution(result=result, input_files=ingested.input_files, artifact_paths=artifact_paths)
