"""The in-process UMUX pipeline result used by reporting adapters.

This module deliberately has no CLI or filesystem orchestration.  It combines
the completed Tasks 1--6 transformations into one immutable result so output
adapters can consume a consistent, reconciled set of dataframes.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class PipelineResult:
    """Final row dispositions and analytical results for one input batch."""

    accepted: pd.DataFrame
    rejected: pd.DataFrame
    monthly_aggregates: pd.DataFrame
    product_summary: pd.DataFrame
    quality: DataQualitySummaries


def run_pipeline(records: pd.DataFrame, config: NormalizationConfig) -> PipelineResult:
    """Run cleaning, duplicate resolution, scoring, and aggregation in order."""
    deduplicated = deduplicate_records(clean_records(records, config))
    accepted = score_accepted_records(deduplicated.accepted)
    monthly_aggregates = build_monthly_aggregates(accepted)
    product_summary = build_product_summaries(monthly_aggregates)
    quality = build_data_quality_summaries(accepted, deduplicated.rejected)
    return PipelineResult(
        accepted=accepted,
        rejected=deduplicated.rejected,
        monthly_aggregates=monthly_aggregates,
        product_summary=product_summary,
        quality=quality,
    )
