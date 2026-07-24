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
        "Панель UMUX-Lite", "Сравнение продуктов и версий", "Тренды UMUX по месяцам",
        "Комбинации с наименьшим последним UMUX", "Наибольшие отрицательные изменения по месяцам",
        "Причины отклонения", "Качество данных", "Как читать эти показатели",
    ]:
        assert heading in html
    assert "Исходные ответы" in html
    assert ">10<" in html
    assert "Средний UMUX</h2><p>62.5</p>" in html
    assert "Plotly.newPlot" in html
    assert '<script src="https://cdn.plot.ly' not in html
    assert "Доля отклонений</h2><p>40.0%</p>" in html


def test_product_version_comparison_uses_all_time_weighted_mean_without_latest_month(tmp_path: Path) -> None:
    html = write_dashboard(_result(), tmp_path, small_sample_threshold=5).read_text(encoding="utf-8")
    comparison = html.split('<section><h2>Сравнение продуктов и версий</h2>', maxsplit=1)[1].split(
        "</section>", maxsplit=1
    )[0]

    assert "<th>Последний месяц</th>" not in comparison
    assert "<th>СРЕДНИЙ UMUX</th>" in comparison
    assert ">66.7<" in comparison
    assert ">50.0<" not in comparison


def test_dashboard_localizes_all_interface_text_to_russian(tmp_path: Path) -> None:
    html = write_dashboard(_result(), tmp_path, small_sample_threshold=5).read_text(encoding="utf-8")

    for text in [
        'lang="ru"', "Панель UMUX-Lite", "Продукт", "Версия", "Ответы",
        "Последний месяц", "СРЕДНИЙ UMUX", "Тренды UMUX по месяцам",
        "Причины отклонения", "Качество данных", "Средний UMUX",
    ]:
        assert text in html
    assert "Product and version comparison" not in html
    assert "Latest month" not in html


def test_dashboard_escapes_hostile_labels_without_small_sample_notice(tmp_path: Path) -> None:
    hostile = '<img src=x onerror="alert(1)">'
    html = write_dashboard(_result(hostile), tmp_path, small_sample_threshold=30).read_text(encoding="utf-8")

    assert hostile not in html
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in html
    assert "\\u003cimg" in html
    assert "Предупреждение о малой выборке" not in html


def test_dashboard_handles_empty_aggregates(tmp_path: Path) -> None:
    empty = _result()
    empty = PipelineResult(
        accepted=pd.DataFrame(), rejected=pd.DataFrame(), monthly_aggregates=pd.DataFrame(),
        product_summary=pd.DataFrame(), quality=empty.quality,
    )

    html = write_dashboard(empty, tmp_path, small_sample_threshold=30).read_text(encoding="utf-8")

    assert "Нет принятых ответов" in html
    assert "Данные о трендах по месяцам отсутствуют" in html


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
    assert "Панель недоступна" in paths["dashboard.html"].read_text(encoding="utf-8")
    assert "Dashboard generation failed" in caplog.text
