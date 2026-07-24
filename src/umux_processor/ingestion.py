"""Deterministic CSV input resolution and schema-level ingestion."""

from __future__ import annotations

import csv
import glob
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_COLUMNS = (
    "response_id",
    "submitted_at",
    "product",
    "product_version",
    "platform",
    "country",
    "user_segment",
    "score1",
    "score2",
)
LINEAGE_COLUMNS = ("source_file", "source_row")
GENERATED_OUTPUT_FILENAMES = frozenset(
    {
        "cleaned_responses.csv",
        "rejected_responses.csv",
        "monthly_aggregates.csv",
        "product_summary.csv",
        "quality_summary.json",
        "dashboard.html",
    }
)


class IngestionError(ValueError):
    """Raised when an input cannot be resolved or read as a usable CSV export."""


@dataclass(frozen=True)
class IngestionResult:
    """Raw schema-valid records, ready for the later cleaning stage.

    ``records`` retains every input column as strings, including columns outside
    the processing schema, and adds ``source_file`` and ``source_row`` lineage.
    No row-level validity decisions or normalization occur at this stage.
    """

    records: pd.DataFrame
    input_files: tuple[Path, ...]


def ingest_csv_inputs(inputs: Sequence[str | Path]) -> IngestionResult:
    """Resolve CSV paths/patterns and combine their schema-valid records.

    Input arguments retain their order. Matches within a glob are sorted by their
    resolved path, and a file matched more than once is loaded only at its first
    occurrence. Reserved generated-artifact names are rejected to avoid feeding a
    previous pipeline output back into processing.
    """

    input_files = _resolve_input_files(inputs)
    frames = [_read_csv_file(path) for path in input_files]
    records = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return IngestionResult(records=records, input_files=input_files)


def _resolve_input_files(inputs: Sequence[str | Path]) -> tuple[Path, ...]:
    if not inputs:
        raise IngestionError("At least one CSV input path or glob pattern is required")

    resolved_files: list[Path] = []
    seen: set[Path] = set()
    for input_value in inputs:
        pattern = str(input_value)
        matches = _resolve_input(pattern)
        for match in matches:
            if match.name in GENERATED_OUTPUT_FILENAMES:
                raise IngestionError(
                    f"Input {match} is a generated output artifact and cannot be ingested"
                )
            if match not in seen:
                seen.add(match)
                resolved_files.append(match)
    return tuple(resolved_files)


def _resolve_input(pattern: str) -> list[Path]:
    matches = [Path(match).resolve() for match in glob.glob(pattern, recursive=True)]
    files = sorted((match for match in matches if match.is_file()), key=lambda path: str(path))
    if not files:
        raise IngestionError(f"Input {pattern!r} did not match any files")
    return files


def _read_csv_file(path: Path) -> pd.DataFrame:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.reader(source, strict=True)
            try:
                header = next(reader)
            except StopIteration as error:
                raise IngestionError(f"File {path}: empty CSV with no usable schema") from error

            _validate_schema(path, header)
            rows, source_rows = _read_rows(path, reader, len(header))
    except IngestionError:
        raise
    except (OSError, UnicodeError) as error:
        raise IngestionError(f"File {path}: could not read CSV: {error}") from error
    except csv.Error as error:
        raise IngestionError(f"File {path}: malformed CSV: {error}") from error

    frame = pd.DataFrame(rows, columns=header, dtype="string")
    frame = _preserve_reserved_extra_columns(frame)
    frame["source_file"] = str(path)
    frame["source_row"] = pd.Series(source_rows, dtype="int64")
    return frame


def _validate_schema(path: Path, header: list[str]) -> None:
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing_columns:
        raise IngestionError(f"File {path}: missing required columns: {', '.join(missing_columns)}")
    extra_columns = [column for column in header if column not in REQUIRED_COLUMNS]
    if extra_columns:
        LOGGER.warning(
            "File %s has extra columns outside the processing schema; retaining them for audit: %s",
            path,
            ", ".join(extra_columns),
        )


def _read_rows(path: Path, reader: csv.reader, expected_columns: int) -> tuple[list[list[str]], list[int]]:
    rows: list[list[str]] = []
    source_rows: list[int] = []
    previous_line = reader.line_num
    for row in reader:
        source_row = previous_line + 1
        previous_line = reader.line_num
        if len(row) != expected_columns:
            raise IngestionError(
                f"File {path}: malformed CSV at line {source_row}: "
                f"expected {expected_columns} columns, found {len(row)}"
            )
        rows.append(row)
        source_rows.append(source_row)
    return rows, source_rows


def _preserve_reserved_extra_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        column: f"input_{column}"
        for column in LINEAGE_COLUMNS
        if column in frame.columns
    }
    if rename_map:
        LOGGER.warning(
            "Renaming input columns that conflict with ingestion lineage: %s",
            ", ".join(rename_map),
        )
        return frame.rename(columns=rename_map)
    return frame
