"""Record normalization and row-level validation.

This stage deliberately neither resolves duplicate response IDs nor calculates
UMUX scores.  It classifies each ingested record exactly once as currently valid
or rejected and retains raw values for audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation

import pandas as pd

from umux_processor.config import NormalizationConfig
from umux_processor.ingestion import LINEAGE_COLUMNS


REASON_COLUMNS = "rejection_reasons"
DUPLICATE_CONTEXT_COLUMN = "duplicate_context"

QUESTIONNAIRE_PAYLOAD_COLUMNS = (
    "submitted_at",
    "product",
    "product_version",
    "platform",
    "country",
    "user_segment",
    "score1",
    "score2",
)


@dataclass(frozen=True)
class CleaningResult:
    """Normalized records split into their current validation status."""

    currently_valid: pd.DataFrame
    rejected: pd.DataFrame

    @property
    def input_count(self) -> int:
        """The number of input records classified by this cleaning run."""
        return len(self.currently_valid) + len(self.rejected)


@dataclass(frozen=True)
class DeduplicationResult:
    """Final cleaning disposition after deterministic duplicate resolution.

    ``duplicate_context`` is audit metadata, distinct from validation reasons.
    Invalid rows retain their original ``rejection_reasons`` and receive one of
    ``duplicate_with_valid``, ``duplicate_conflict``, or
    ``duplicate_all_invalid`` when another row shares their normalized ID.
    Valid copies rejected by deduplication instead receive the corresponding
    ``duplicate_exact`` or ``duplicate_conflict`` rejection reason.
    """

    accepted: pd.DataFrame
    rejected: pd.DataFrame

    @property
    def input_count(self) -> int:
        """The number of rows accounted for after deduplication."""
        return len(self.accepted) + len(self.rejected)


def clean_records(records: pd.DataFrame, config: NormalizationConfig) -> CleaningResult:
    """Normalize and validate raw ingested records without mutating ``records``.

    All non-lineage input columns are copied to ``original_<column>`` before
    transformation. Scores use an explicit lexical policy: finite numeric values
    that are exactly integral are accepted (``3``, ``3.0``, and ``3.00`` all
    become integer ``3``); fractional and nonnumeric values are rejected.
    """
    normalized = records.copy(deep=True)
    for column in records.columns:
        if column not in LINEAGE_COLUMNS:
            normalized[f"original_{column}"] = records[column]

    product_aliases = _normalized_aliases(config.product_aliases)
    platform_aliases = {platform.casefold(): platform for platform in config.supported_platforms}
    segment_aliases = _normalized_aliases(config.user_segment_aliases)

    reasons_by_row: list[tuple[str, ...]] = []
    transformed: list[dict[str, object]] = []
    for _, row in normalized.iterrows():
        values, reasons = _clean_row(row, config, product_aliases, platform_aliases, segment_aliases)
        transformed.append(values)
        reasons_by_row.append(tuple(reasons))

    for column in ("response_id", "submitted_at", "product", "product_version", "platform", "country", "user_segment", "score1", "score2"):
        normalized[column] = [values[column] for values in transformed]
    normalized[REASON_COLUMNS] = reasons_by_row

    valid_mask = normalized[REASON_COLUMNS].map(len).eq(0)
    return CleaningResult(
        currently_valid=normalized.loc[valid_mask].drop(columns=REASON_COLUMNS).reset_index(drop=True),
        rejected=normalized.loc[~valid_mask].reset_index(drop=True),
    )


def deduplicate_records(cleaned: CleaningResult) -> DeduplicationResult:
    """Resolve duplicate normalized response IDs without changing row validity.

    A group is ordered by input argument order (when ingestion supplied
    ``source_input_order``), then ``source_file`` and ``source_row``.  For
    callers providing only the documented file/row lineage, file and row order
    remains deterministic even if dataframe rows are shuffled.
    """
    valid = cleaned.currently_valid.copy(deep=True)
    invalid = cleaned.rejected.copy(deep=True)
    valid[DUPLICATE_CONTEXT_COLUMN] = pd.NA
    invalid[DUPLICATE_CONTEXT_COLUMN] = pd.NA

    accepted_parts: list[pd.DataFrame] = []
    rejected_parts: list[pd.DataFrame] = []
    valid_by_id = _records_by_response_id(valid)
    invalid_by_id = _records_by_response_id(invalid)

    response_ids = sorted(set(valid_by_id) | set(invalid_by_id))
    for response_id in response_ids:
        valid_group = valid_by_id.get(response_id, valid.iloc[0:0])
        invalid_group = invalid_by_id.get(response_id, invalid.iloc[0:0])
        if valid_group.empty:
            invalid_group = invalid_group.copy()
            invalid_group[DUPLICATE_CONTEXT_COLUMN] = "duplicate_all_invalid"
            rejected_parts.append(invalid_group)
            continue

        payload_count = len(_payload_keys(valid_group))
        if payload_count == 1:
            ordered_valid = _sort_by_lineage(valid_group)
            accepted_parts.append(ordered_valid.iloc[[0]])
            duplicate_valid = ordered_valid.iloc[1:].copy()
            if not duplicate_valid.empty:
                duplicate_valid[REASON_COLUMNS] = [("duplicate_exact",)] * len(duplicate_valid)
                duplicate_valid[DUPLICATE_CONTEXT_COLUMN] = "duplicate_exact"
                rejected_parts.append(duplicate_valid)
            invalid_context = "duplicate_with_valid"
        else:
            conflicted_valid = valid_group.copy()
            conflicted_valid[REASON_COLUMNS] = [("duplicate_conflict",)] * len(conflicted_valid)
            conflicted_valid[DUPLICATE_CONTEXT_COLUMN] = "duplicate_conflict"
            rejected_parts.append(conflicted_valid)
            invalid_context = "duplicate_conflict"

        if not invalid_group.empty:
            invalid_group = invalid_group.copy()
            invalid_group[DUPLICATE_CONTEXT_COLUMN] = invalid_context
            rejected_parts.append(invalid_group)

    missing_id_invalid = invalid.loc[invalid["response_id"].isna()]
    if not missing_id_invalid.empty:
        rejected_parts.append(missing_id_invalid)

    accepted = _combine_and_sort(accepted_parts, valid)
    rejected = _combine_and_sort(rejected_parts, invalid)
    return DeduplicationResult(accepted=accepted, rejected=rejected)


def _records_by_response_id(records: pd.DataFrame) -> dict[str, pd.DataFrame]:
    identified = records.loc[records["response_id"].notna()]
    return {str(response_id): group.copy() for response_id, group in identified.groupby("response_id", sort=False)}


def _payload_keys(records: pd.DataFrame) -> set[tuple[object, ...]]:
    return {tuple(row) for row in records.loc[:, QUESTIONNAIRE_PAYLOAD_COLUMNS].itertuples(index=False, name=None)}


def _sort_by_lineage(records: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in ("source_input_order", "source_file", "source_row") if column in records]
    return records.sort_values(columns, kind="stable") if columns else records.copy()


def _combine_and_sort(parts: list[pd.DataFrame], template: pd.DataFrame) -> pd.DataFrame:
    if not parts:
        return template.iloc[0:0].copy().reset_index(drop=True)
    return _sort_by_lineage(pd.concat(parts, ignore_index=True, sort=False)).reset_index(drop=True)


def _clean_row(
    row: pd.Series,
    config: NormalizationConfig,
    product_aliases: dict[str, str],
    platform_aliases: dict[str, str],
    segment_aliases: dict[str, str],
) -> tuple[dict[str, object], list[str]]:
    reasons: list[str] = []
    response_id = _string_or_none(row["response_id"])
    if response_id is None:
        reasons.append("missing_response_id")

    submitted_at = _parse_timestamp(row["submitted_at"], config.timestamp_format)
    if submitted_at is None:
        reasons.append("invalid_submitted_at")

    product_input = _string_or_none(row["product"])
    if product_input is None:
        product = None
        reasons.append("missing_product")
    else:
        product = product_aliases.get(product_input.casefold())
        if product is None:
            reasons.append("unknown_product")

    product_version = _string_or_none(row["product_version"])
    if product_version is None:
        reasons.append("missing_product_version")

    platform = _optional_alias(row["platform"], platform_aliases, "Unknown")
    country = _normalize_country(row["country"], config)
    user_segment = _optional_alias(row["user_segment"], segment_aliases, "Unknown")

    score1 = _parse_score(row["score1"], "score1", reasons)
    score2 = _parse_score(row["score2"], "score2", reasons)
    return (
        {
            "response_id": response_id,
            "submitted_at": submitted_at,
            "product": product,
            "product_version": product_version,
            "platform": platform,
            "country": country,
            "user_segment": user_segment,
            "score1": score1,
            "score2": score2,
        },
        reasons,
    )


def _normalized_aliases(aliases: object) -> dict[str, str]:
    return {str(alias).strip().casefold(): canonical for alias, canonical in aliases.items()}  # type: ignore[union-attr]


def _string_or_none(value: object) -> str | None:
    if value is None or _is_missing(value):
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    return value or None


def _is_missing(value: object) -> bool:
    missing = pd.isna(value)
    return bool(missing) if not isinstance(missing, (list, tuple)) else False


def _parse_timestamp(value: object, timestamp_format: str) -> datetime | None:
    text = _string_or_none(value)
    if text is None:
        return None
    try:
        return datetime.strptime(text, timestamp_format)
    except ValueError:
        return None


def _optional_alias(value: object, aliases: dict[str, str], missing_value: str) -> str:
    text = _string_or_none(value)
    return missing_value if text is None else aliases.get(text.casefold(), missing_value)


def _normalize_country(value: object, config: NormalizationConfig) -> str:
    text = _string_or_none(value)
    if text is None:
        return config.countries.missing_value
    normalized = text.upper() if config.countries.case == "upper" else text
    if len(normalized) != config.countries.code_length or not normalized.isalpha():
        return config.countries.missing_value
    return normalized


def _parse_score(value: object, field: str, reasons: list[str]) -> int | None:
    text = _string_or_none(value)
    if text is None:
        reasons.append(f"missing_{field}")
        return None
    try:
        decimal = Decimal(text)
    except (InvalidOperation, ValueError):
        reasons.append(f"non_integer_{field}")
        return None
    if not decimal.is_finite() or decimal != decimal.to_integral_value():
        reasons.append(f"non_integer_{field}")
        return None
    score = int(decimal)
    if not 1 <= score <= 5:
        reasons.append(f"{field}_out_of_range")
        return None
    return score
