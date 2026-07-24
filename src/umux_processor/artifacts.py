"""Deterministic, machine-readable audit artifact generation.

All artifacts are serialized to temporary files in the requested output
directory before any final artifact is replaced.  This prevents a
serialization failure from changing an existing artifact set.  Replacing the
five final paths is necessarily five separate atomic filesystem operations,
so an OS failure during that final replacement phase can leave an old/new
mixture; cross-file atomicity is not achievable with independent files.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Callable

import pandas as pd

from umux_processor.aggregation import MONTHLY_COLUMNS, PRODUCT_SUMMARY_COLUMNS
from umux_processor.pipeline import PipelineResult


ARTIFACT_FILENAMES = (
    "cleaned_responses.csv",
    "rejected_responses.csv",
    "monthly_aggregates.csv",
    "product_summary.csv",
    "quality_summary.json",
)
CLEANED_RESPONSE_COLUMNS = [
    "response_id", "submitted_at", "product", "product_version", "platform",
    "country", "user_segment", "score1", "score2", "umux_score", "source_file",
    "source_row", "source_input_order",
]
REJECTED_RESPONSE_COLUMNS = [
    "source_file", "source_row", "source_input_order", "original_response_id",
    "original_submitted_at", "original_product", "original_product_version",
    "original_platform", "original_country", "original_user_segment", "original_score1",
    "original_score2", "rejection_reasons", "duplicate_context",
]
MONTHLY_AGGREGATE_ARTIFACT_COLUMNS = MONTHLY_COLUMNS
PRODUCT_SUMMARY_ARTIFACT_COLUMNS = PRODUCT_SUMMARY_COLUMNS


class ArtifactWriteError(RuntimeError):
    """Raised when audit artifacts could not be safely prepared or replaced."""


def write_audit_artifacts(result: PipelineResult, output_directory: str | Path) -> dict[str, Path]:
    """Replace known audit artifacts in ``output_directory`` deterministically.

    The directory is created if absent but is never cleared.  Each final file
    is replaced only after every artifact has serialized successfully to a
    temporary sibling file.
    """
    directory = Path(output_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ArtifactWriteError(f"Could not create output directory {directory}: {error}") from error

    temporary_paths: dict[str, Path] = {}
    try:
        temporary_paths["cleaned_responses.csv"] = _write_csv_temp(
            directory, _cleaned_artifact(result.accepted), CLEANED_RESPONSE_COLUMNS
        )
        temporary_paths["rejected_responses.csv"] = _write_csv_temp(
            directory, _rejected_artifact(result.rejected), REJECTED_RESPONSE_COLUMNS
        )
        temporary_paths["monthly_aggregates.csv"] = _write_csv_temp(
            directory, _tabular_artifact(result.monthly_aggregates, MONTHLY_AGGREGATE_ARTIFACT_COLUMNS),
            MONTHLY_AGGREGATE_ARTIFACT_COLUMNS,
        )
        temporary_paths["product_summary.csv"] = _write_csv_temp(
            directory, _tabular_artifact(result.product_summary, PRODUCT_SUMMARY_ARTIFACT_COLUMNS),
            PRODUCT_SUMMARY_ARTIFACT_COLUMNS,
        )
        temporary_paths["quality_summary.json"] = _write_json_temp(directory, _quality_summary(result))
    except Exception as error:
        artifact_name = next(name for name in ARTIFACT_FILENAMES if name not in temporary_paths)
        raise ArtifactWriteError(f"Could not serialize {artifact_name}: {error}") from error
    finally:
        if len(temporary_paths) != len(ARTIFACT_FILENAMES):
            _remove_temporary_files(temporary_paths.values())

    try:
        for filename in ARTIFACT_FILENAMES:
            os.replace(temporary_paths[filename], directory / filename)
    except OSError as error:
        _remove_temporary_files(temporary_paths.values())
        raise ArtifactWriteError(f"Could not replace audit artifact: {error}") from error
    return {filename: directory / filename for filename in ARTIFACT_FILENAMES}


def _cleaned_artifact(accepted: pd.DataFrame) -> pd.DataFrame:
    return _tabular_artifact(accepted, CLEANED_RESPONSE_COLUMNS)


def _rejected_artifact(rejected: pd.DataFrame) -> pd.DataFrame:
    artifact = _tabular_artifact(rejected, REJECTED_RESPONSE_COLUMNS)
    artifact["rejection_reasons"] = artifact["rejection_reasons"].map(_serialize_reasons)
    return artifact


def _tabular_artifact(records: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    artifact = records.reindex(columns=columns).copy()
    for column in artifact.columns:
        if pd.api.types.is_datetime64_any_dtype(artifact[column]):
            artifact[column] = artifact[column].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return artifact


def _quality_summary(result: PipelineResult) -> dict[str, object]:
    overall = result.quality.overall.iloc[0]
    reason_counts = {
        str(row.rejection_reason): int(row.rejected_row_count)
        for row in result.quality.by_rejection_reason.itertuples(index=False)
    }
    duplicate_count = sum(
        count for reason, count in reason_counts.items()
        if reason in {"duplicate_exact", "duplicate_conflict"}
    )
    return {
        "accepted_row_count": int(overall["accepted_row_count"]),
        "duplicate_row_count": duplicate_count,
        "raw_row_count": int(overall["raw_row_count"]),
        "rejected_row_count": int(overall["rejected_row_count"]),
        "rejection_reason_counts": reason_counts,
    }


def _serialize_reasons(value: object) -> str:
    if isinstance(value, (tuple, list)):
        return json.dumps([str(reason) for reason in value], ensure_ascii=False, separators=(",", ":"))
    return json.dumps([], separators=(",", ":"))


def _write_csv_temp(directory: Path, records: pd.DataFrame, columns: list[str]) -> Path:
    return _write_temp(directory, ".csv", lambda handle: records.to_csv(
        handle, index=False, columns=columns, lineterminator="\n", date_format="%Y-%m-%dT%H:%M:%S"
    ))


def _write_json_temp(directory: Path, content: dict[str, object]) -> Path:
    return _write_temp(directory, ".json", lambda handle: json.dump(
        content, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))


def _write_temp(directory: Path, suffix: str, serialize: Callable[[object], None]) -> Path:
    descriptor, filename = tempfile.mkstemp(prefix=".umux-artifact-", suffix=suffix, dir=directory, text=True)
    path = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            serialize(handle)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _remove_temporary_files(paths: object) -> None:
    for path in paths:  # type: ignore[union-attr]
        Path(path).unlink(missing_ok=True)
