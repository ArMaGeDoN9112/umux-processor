"""Deterministic analytical aggregations for scored UMUX-Lite records."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


MONTHLY_COLUMNS = [
    "month",
    "product",
    "product_version",
    "valid_responses",
    "mean_umux",
    "median_umux",
    "previous_month_mean",
    "month_over_month_delta",
]
PRODUCT_SUMMARY_COLUMNS = [
    "product",
    "total_valid_responses",
    "overall_mean_umux",
    "latest_month",
    "latest_month_mean_umux",
    "previous_calendar_month_mean_umux",
    "latest_month_over_month_delta",
]
QUALITY_OVERALL_COLUMNS = ["raw_row_count", "accepted_row_count", "rejected_row_count", "rejection_rate"]
QUALITY_REASON_COLUMNS = ["rejection_reason", "rejected_row_count"]


@dataclass(frozen=True)
class DataQualitySummaries:
    """Global and usable-dimension data-quality summaries.

    ``by_rejection_reason`` counts every reason attached to a rejected row, so
    its counts can sum to more than the rejected-row count when a row has
    multiple validation failures.
    """

    overall: pd.DataFrame
    by_rejection_reason: pd.DataFrame
    by_product: pd.DataFrame
    by_source: pd.DataFrame


def build_monthly_aggregates(scored_accepted: pd.DataFrame) -> pd.DataFrame:
    """Aggregate accepted scored records by calendar month, product, and version.

    ``submitted_at`` is the parsed, timezone-naive timestamp from cleaning.
    Changes are populated only when the immediately preceding calendar month
    exists for the same product/version combination.
    """
    _require_columns(scored_accepted, {"submitted_at", "product", "product_version", "umux_score"})
    if scored_accepted.empty:
        return pd.DataFrame(columns=MONTHLY_COLUMNS)

    records = scored_accepted.loc[:, ["submitted_at", "product", "product_version", "umux_score"]].copy()
    records["month"] = pd.to_datetime(records["submitted_at"]).dt.to_period("M").dt.to_timestamp()
    grouped = (
        records.groupby(["month", "product", "product_version"], sort=True, dropna=False)["umux_score"]
        .agg(valid_responses="count", mean_umux="mean", median_umux="median")
        .reset_index()
        .sort_values(["month", "product", "product_version"], kind="stable")
        .reset_index(drop=True)
    )

    previous_means: list[float | None] = []
    deltas: list[float | None] = []
    for _, group in grouped.groupby(["product", "product_version"], sort=False, dropna=False):
        prior_month: pd.Timestamp | None = None
        prior_mean: float | None = None
        for row in group.itertuples(index=False):
            current_month = row.month
            current_mean = float(row.mean_umux)
            if prior_month is not None and current_month == prior_month + pd.DateOffset(months=1):
                previous_means.append(prior_mean)
                deltas.append(current_mean - prior_mean)  # type: ignore[operator]
            else:
                previous_means.append(None)
                deltas.append(None)
            prior_month = current_month
            prior_mean = current_mean

    grouped["previous_month_mean"] = pd.Series(previous_means, dtype="object")
    grouped["month_over_month_delta"] = pd.Series(deltas, dtype="object")
    return grouped.loc[:, MONTHLY_COLUMNS]


def build_product_summaries(monthly_aggregates: pd.DataFrame) -> pd.DataFrame:
    """Summarize product performance, weighted by monthly sample sizes."""
    _require_columns(monthly_aggregates, {"month", "product", "valid_responses", "mean_umux"})
    if monthly_aggregates.empty:
        return pd.DataFrame(columns=PRODUCT_SUMMARY_COLUMNS)

    monthly = monthly_aggregates.loc[:, ["month", "product", "valid_responses", "mean_umux"]].copy()
    monthly["month"] = pd.to_datetime(monthly["month"])
    rows: list[dict[str, object]] = []
    for product, product_rows in monthly.groupby("product", sort=True, dropna=False):
        by_month = product_rows.groupby("month", sort=True).apply(_combine_month, include_groups=False)
        latest_month = by_month.index[-1]
        latest_mean = float(by_month.loc[latest_month, "mean_umux"])
        previous_month = latest_month - pd.DateOffset(months=1)
        if previous_month in by_month.index:
            previous_mean: float | None = float(by_month.loc[previous_month, "mean_umux"])
            delta: float | None = latest_mean - previous_mean
        else:
            previous_mean = None
            delta = None
        total_responses = int(by_month["valid_responses"].sum())
        overall_mean = float((by_month["valid_responses"] * by_month["mean_umux"]).sum() / total_responses)
        rows.append(
            {
                "product": product,
                "total_valid_responses": total_responses,
                "overall_mean_umux": overall_mean,
                "latest_month": latest_month,
                "latest_month_mean_umux": latest_mean,
                "previous_calendar_month_mean_umux": previous_mean,
                "latest_month_over_month_delta": delta,
            }
        )
    result = pd.DataFrame(rows, columns=PRODUCT_SUMMARY_COLUMNS).sort_values("product", kind="stable").reset_index(drop=True)
    for column in ("previous_calendar_month_mean_umux", "latest_month_over_month_delta"):
        result[column] = result[column].astype("object").where(result[column].notna(), None)
    return result


def build_data_quality_summaries(accepted: pd.DataFrame, rejected: pd.DataFrame) -> DataQualitySummaries:
    """Return global, rejection-reason, product, and source quality totals.

    Both inputs are the final deduplication dispositions, with ``accepted``
    scored by Task 5. Their combined length is the raw ingested row count,
    including discarded duplicate copies, which is the rejection-rate
    denominator.
    """
    accepted_count = len(accepted)
    rejected_count = len(rejected)
    raw_count = accepted_count + rejected_count
    overall = pd.DataFrame(
        [{
            "raw_row_count": raw_count,
            "accepted_row_count": accepted_count,
            "rejected_row_count": rejected_count,
            "rejection_rate": rejected_count / raw_count if raw_count else 0.0,
        }],
        columns=QUALITY_OVERALL_COLUMNS,
    )
    return DataQualitySummaries(
        overall=overall,
        by_rejection_reason=_rejection_reason_counts(rejected),
        by_product=_quality_by_dimension(accepted, rejected, "product"),
        by_source=_quality_by_dimension(accepted, rejected, "source_file"),
    )


def _combine_month(group: pd.DataFrame) -> pd.Series:
    valid_responses = int(group["valid_responses"].sum())
    return pd.Series(
        {
            "valid_responses": valid_responses,
            "mean_umux": (group["valid_responses"] * group["mean_umux"]).sum() / valid_responses,
        }
    )


def _rejection_reason_counts(rejected: pd.DataFrame) -> pd.DataFrame:
    if "rejection_reasons" not in rejected or rejected.empty:
        return pd.DataFrame(columns=QUALITY_REASON_COLUMNS)
    reasons = [reason for values in rejected["rejection_reasons"] for reason in _reasons(values)]
    if not reasons:
        return pd.DataFrame(columns=QUALITY_REASON_COLUMNS)
    return (
        pd.Series(reasons, name="rejection_reason")
        .value_counts(sort=False)
        .rename_axis("rejection_reason")
        .reset_index(name="rejected_row_count")
        .sort_values("rejection_reason", kind="stable")
        .reset_index(drop=True)
    )


def _quality_by_dimension(accepted: pd.DataFrame, rejected: pd.DataFrame, dimension: str) -> pd.DataFrame:
    columns = [dimension, *QUALITY_OVERALL_COLUMNS]
    if dimension not in accepted.columns and dimension not in rejected.columns:
        return pd.DataFrame(columns=columns)
    accepted_values = _usable_dimension_rows(accepted, dimension, "accepted")
    rejected_values = _usable_dimension_rows(rejected, dimension, "rejected")
    records = pd.concat([accepted_values, rejected_values], ignore_index=True)
    if records.empty:
        return pd.DataFrame(columns=columns)
    summary = records.groupby(dimension, sort=True)["disposition"].value_counts().unstack(fill_value=0)
    summary["raw_row_count"] = summary.sum(axis=1)
    summary["accepted_row_count"] = summary.get("accepted", 0)
    summary["rejected_row_count"] = summary.get("rejected", 0)
    summary["rejection_rate"] = summary["rejected_row_count"] / summary["raw_row_count"]
    return summary.reset_index().loc[:, columns].sort_values(dimension, kind="stable").reset_index(drop=True)


def _usable_dimension_rows(records: pd.DataFrame, dimension: str, disposition: str) -> pd.DataFrame:
    if dimension not in records.columns:
        return pd.DataFrame(columns=[dimension, "disposition"])
    values = records.loc[:, [dimension]].copy()
    usable = values[dimension].notna() & values[dimension].astype(str).str.strip().ne("")
    values = values.loc[usable]
    values["disposition"] = disposition
    return values


def _reasons(value: object) -> tuple[str, ...]:
    if isinstance(value, (tuple, list)):
        return tuple(str(reason) for reason in value)
    return ()


def _require_columns(records: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required.difference(records.columns))
    if missing:
        raise ValueError(f"Records are missing required aggregation columns: {', '.join(missing)}")
