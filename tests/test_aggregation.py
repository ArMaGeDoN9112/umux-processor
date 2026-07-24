"""Hand-calculated contracts for Task 6 analytical aggregation."""

from __future__ import annotations

import pandas as pd

from umux_processor.aggregation import (
    build_data_quality_summaries,
    build_monthly_aggregates,
    build_product_summaries,
)


def test_monthly_aggregates_calculate_metrics_for_adjacent_months_and_sort_rows() -> None:
    scored = pd.DataFrame(
        [
            {"submitted_at": pd.Timestamp("2024-03-02 08:00:00"), "product": "Search", "product_version": "2.0", "umux_score": 100.0},
            {"submitted_at": pd.Timestamp("2024-01-10 08:00:00"), "product": "Payments", "product_version": "10", "umux_score": 50.0},
            {"submitted_at": pd.Timestamp("2024-02-03 08:00:00"), "product": "Payments", "product_version": "2.0", "umux_score": 0.0},
            {"submitted_at": pd.Timestamp("2024-01-03 08:00:00"), "product": "Payments", "product_version": "2.0", "umux_score": 50.0},
            {"submitted_at": pd.Timestamp("2024-01-04 08:00:00"), "product": "Payments", "product_version": "2.0", "umux_score": 100.0},
            {"submitted_at": pd.Timestamp("2024-03-01 08:00:00"), "product": "Search", "product_version": "2.0", "umux_score": 50.0},
        ]
    )

    result = build_monthly_aggregates(scored)

    assert result.columns.tolist() == [
        "month",
        "product",
        "product_version",
        "valid_responses",
        "mean_umux",
        "median_umux",
        "previous_month_mean",
        "month_over_month_delta",
    ]
    assert result.to_dict("records") == [
        {"month": pd.Timestamp("2024-01-01"), "product": "Payments", "product_version": "10", "valid_responses": 1, "mean_umux": 50.0, "median_umux": 50.0, "previous_month_mean": None, "month_over_month_delta": None},
        {"month": pd.Timestamp("2024-01-01"), "product": "Payments", "product_version": "2.0", "valid_responses": 2, "mean_umux": 75.0, "median_umux": 75.0, "previous_month_mean": None, "month_over_month_delta": None},
        {"month": pd.Timestamp("2024-02-01"), "product": "Payments", "product_version": "2.0", "valid_responses": 1, "mean_umux": 0.0, "median_umux": 0.0, "previous_month_mean": 75.0, "month_over_month_delta": -75.0},
        {"month": pd.Timestamp("2024-03-01"), "product": "Search", "product_version": "2.0", "valid_responses": 2, "mean_umux": 75.0, "median_umux": 75.0, "previous_month_mean": None, "month_over_month_delta": None},
    ]
    assert result["valid_responses"].sum() == len(scored)


def test_product_summaries_use_latest_calendar_month_and_exclude_non_adjacent_history() -> None:
    monthly = pd.DataFrame(
        [
            {"month": pd.Timestamp("2024-03-01"), "product": "Search", "product_version": "2.0", "valid_responses": 2, "mean_umux": 75.0, "median_umux": 75.0, "previous_month_mean": None, "month_over_month_delta": None},
            {"month": pd.Timestamp("2024-01-01"), "product": "Payments", "product_version": "2.0", "valid_responses": 2, "mean_umux": 75.0, "median_umux": 75.0, "previous_month_mean": None, "month_over_month_delta": None},
            {"month": pd.Timestamp("2024-02-01"), "product": "Payments", "product_version": "2.0", "valid_responses": 1, "mean_umux": 0.0, "median_umux": 0.0, "previous_month_mean": 75.0, "month_over_month_delta": -75.0},
            {"month": pd.Timestamp("2024-01-01"), "product": "Payments", "product_version": "10", "valid_responses": 1, "mean_umux": 50.0, "median_umux": 50.0, "previous_month_mean": None, "month_over_month_delta": None},
        ]
    )

    result = build_product_summaries(monthly)

    assert result.columns.tolist() == [
        "product",
        "total_valid_responses",
        "overall_mean_umux",
        "latest_month",
        "latest_month_mean_umux",
        "previous_calendar_month_mean_umux",
        "latest_month_over_month_delta",
    ]
    assert result.to_dict("records") == [
        {"product": "Payments", "total_valid_responses": 4, "overall_mean_umux": 50.0, "latest_month": pd.Timestamp("2024-02-01"), "latest_month_mean_umux": 0.0, "previous_calendar_month_mean_umux": 66.66666666666667, "latest_month_over_month_delta": -66.66666666666667},
        {"product": "Search", "total_valid_responses": 2, "overall_mean_umux": 75.0, "latest_month": pd.Timestamp("2024-03-01"), "latest_month_mean_umux": 75.0, "previous_calendar_month_mean_umux": None, "latest_month_over_month_delta": None},
    ]


def test_quality_summaries_count_all_reasons_and_keep_rejected_rows_out_of_scores() -> None:
    accepted = pd.DataFrame(
        [
            {"submitted_at": pd.Timestamp("2024-01-02 08:00:00"), "product": "Payments", "product_version": "2.0", "source_file": "a.csv", "umux_score": 100.0},
            {"submitted_at": pd.Timestamp("2024-01-02 08:00:00"), "product": "Search", "product_version": "2.0", "source_file": "a.csv", "umux_score": 50.0},
        ]
    )
    rejected = pd.DataFrame(
        [
            {"product": "Payments", "source_file": "a.csv", "umux_score": 0.0, "rejection_reasons": ("missing_score1", "invalid_submitted_at")},
            {"product": "Payments", "source_file": "b.csv", "rejection_reasons": ("duplicate_exact",)},
            {"product": None, "source_file": "b.csv", "rejection_reasons": ("missing_product",)},
        ]
    )

    result = build_data_quality_summaries(accepted, rejected)
    monthly = build_monthly_aggregates(accepted)

    assert monthly[["product", "mean_umux"]].to_dict("records") == [
        {"product": "Payments", "mean_umux": 100.0},
        {"product": "Search", "mean_umux": 50.0},
    ]
    assert result.overall.to_dict("records") == [
        {"raw_row_count": 5, "accepted_row_count": 2, "rejected_row_count": 3, "rejection_rate": 0.6}
    ]
    assert result.by_rejection_reason.to_dict("records") == [
        {"rejection_reason": "duplicate_exact", "rejected_row_count": 1},
        {"rejection_reason": "invalid_submitted_at", "rejected_row_count": 1},
        {"rejection_reason": "missing_product", "rejected_row_count": 1},
        {"rejection_reason": "missing_score1", "rejected_row_count": 1},
    ]
    assert result.by_product.to_dict("records") == [
        {"product": "Payments", "raw_row_count": 3, "accepted_row_count": 1, "rejected_row_count": 2, "rejection_rate": 2 / 3},
        {"product": "Search", "raw_row_count": 1, "accepted_row_count": 1, "rejected_row_count": 0, "rejection_rate": 0.0},
    ]
    assert result.by_source.to_dict("records") == [
        {"source_file": "a.csv", "raw_row_count": 3, "accepted_row_count": 2, "rejected_row_count": 1, "rejection_rate": 1 / 3},
        {"source_file": "b.csv", "raw_row_count": 2, "accepted_row_count": 0, "rejected_row_count": 2, "rejection_rate": 1.0},
    ]


def test_aggregations_handle_empty_and_single_month_inputs() -> None:
    empty_scored = pd.DataFrame(columns=["submitted_at", "product", "product_version", "umux_score"])
    empty_monthly = build_monthly_aggregates(empty_scored)

    assert empty_monthly.empty
    assert build_product_summaries(empty_monthly).empty

    quality = build_data_quality_summaries(pd.DataFrame(), pd.DataFrame(columns=["rejection_reasons"]))
    assert quality.overall.to_dict("records") == [
        {"raw_row_count": 0, "accepted_row_count": 0, "rejected_row_count": 0, "rejection_rate": 0.0}
    ]
    assert quality.by_rejection_reason.empty
    assert quality.by_product.empty
    assert quality.by_source.empty
