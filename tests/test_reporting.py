from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from umux_processor.aggregation import DataQualitySummaries
from umux_processor.artifacts import write_audit_artifacts
from umux_processor.pipeline import PipelineResult
from umux_processor.reporting import write_dashboard


def _result(product: str = "Payments") -> PipelineResult:
    monthly = pd.DataFrame(
        [
            {"month": pd.Timestamp("2024-01-01"), "product": product, "product_version": "2.0", "valid_responses": 4, "mean_umux": 75.0, "median_umux": 75.0, "previous_month_mean": None, "month_over_month_delta": None},
            {"month": pd.Timestamp("2024-02-01"), "product": product, "product_version": "2.0", "valid_responses": 2, "mean_umux": 50.0, "median_umux": 50.0, "previous_month_mean": 75.0, "month_over_month_delta": -25.0},
        ]
    )
    quality = DataQualitySummaries(
        overall=pd.DataFrame([{"raw_row_count": 10, "accepted_row_count": 6, "rejected_row_count": 4, "rejection_rate": 0.4}]),
        by_rejection_reason=pd.DataFrame([{"rejection_reason": "missing_score1", "rejected_row_count": 3}]),
        by_product=pd.DataFrame([{"product": product, "raw_row_count": 10, "accepted_row_count": 6, "rejected_row_count": 4, "rejection_rate": 0.4}]),
        by_source=pd.DataFrame([{"source_file": "responses.csv", "raw_row_count": 10, "accepted_row_count": 6, "rejected_row_count": 4, "rejection_rate": 0.4}]),
    )
    return PipelineResult(
        accepted=pd.DataFrame(), rejected=pd.DataFrame(), monthly_aggregates=monthly,
        product_summary=pd.DataFrame([{"product": product, "total_valid_responses": 6, "overall_mean_umux": 62.5}]), quality=quality,
    )


def test_dashboard_contains_sections_embedded_assets_and_provided_values(tmp_path: Path) -> None:
    path = write_dashboard(_result(), tmp_path, small_sample_threshold=5)
    html = path.read_text(encoding="utf-8")

    for heading in [
        "UMUX-Lite dashboard", "Product and version comparison", "Monthly UMUX trends",
        "Lowest-performing latest combinations", "Largest negative calendar-month changes",
        "Rejection reasons", "Data quality", "How to read these metrics",
    ]:
        assert heading in html
    assert "Raw responses" in html
    assert ">10<" in html
    assert "Overall mean UMUX</h2><p>62.5</p>" in html
    assert "Plotly.newPlot" in html
    assert '<script src="https://cdn.plot.ly' not in html
    assert "Rejection rate</h2><p>40.0%</p>" in html


def test_dashboard_escapes_hostile_labels_and_warns_about_small_samples(tmp_path: Path) -> None:
    hostile = '<img src=x onerror="alert(1)">'
    html = write_dashboard(_result(hostile), tmp_path, small_sample_threshold=30).read_text(encoding="utf-8")

    assert hostile not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "\\u003cimg" in html
    assert "Small-sample notice" in html
    assert "below the configured threshold of 30" in html


def test_dashboard_handles_empty_aggregates(tmp_path: Path) -> None:
    empty = _result()
    empty = PipelineResult(
        accepted=pd.DataFrame(), rejected=pd.DataFrame(), monthly_aggregates=pd.DataFrame(),
        product_summary=pd.DataFrame(), quality=empty.quality,
    )

    html = write_dashboard(empty, tmp_path, small_sample_threshold=30).read_text(encoding="utf-8")

    assert "No accepted responses are available" in html
    assert "No monthly trend data is available" in html


def test_dashboard_failure_does_not_prevent_machine_readable_artifacts(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    def fail_dashboard(*args: object, **kwargs: object) -> Path:
        raise RuntimeError("simulated Plotly failure")

    monkeypatch.setattr("umux_processor.artifacts.write_dashboard", fail_dashboard)
    with caplog.at_level(logging.ERROR):
        paths = write_audit_artifacts(_result(), tmp_path, small_sample_threshold=30)

    assert (tmp_path / "quality_summary.json").exists()
    assert paths["dashboard.html"].exists()
    assert "Dashboard unavailable" in paths["dashboard.html"].read_text(encoding="utf-8")
    assert "Dashboard generation failed" in caplog.text
