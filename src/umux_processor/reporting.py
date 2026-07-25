"""Standalone, escaped HTML dashboard rendering from pipeline aggregates."""

from __future__ import annotations

import html
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd
import plotly.graph_objects as go
from _plotly_utils.utils import PlotlyJSONEncoder
from plotly.offline.offline import get_plotlyjs

from umux_processor.pipeline import PipelineResult


def write_dashboard(
    result: PipelineResult, output_directory: str | Path, *, small_sample_threshold: int
) -> Path:
    """Write a self-contained dashboard using already-calculated pipeline outputs."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dashboard.html"
    descriptor, filename = tempfile.mkstemp(prefix=".umux-dashboard-", suffix=".html", dir=directory, text=True)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(_dashboard_html(result, small_sample_threshold))
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def write_dashboard_failure_notice(output_directory: str | Path) -> Path:
    """Replace a stale report with a safe notice when dashboard rendering fails."""
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "dashboard.html"
    descriptor, filename = tempfile.mkstemp(prefix=".umux-dashboard-", suffix=".html", dir=directory, text=True)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write("<!doctype html><html lang=\"ru\"><meta charset=\"utf-8\"><title>Панель недоступна</title><body><h1>Панель недоступна</h1><p>Не удалось сформировать HTML-панель. CSV- и JSON-артефакты аудита созданы успешно; подробности приведены в журналах конвейера.</p></body></html>")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _dashboard_html(result: PipelineResult, threshold: int) -> str:
    overall = result.quality.overall.iloc[0] if not result.quality.overall.empty else {}
    raw = int(overall.get("raw_row_count", 0))
    accepted = int(overall.get("accepted_row_count", 0))
    rejection_rate = float(overall.get("rejection_rate", 0.0))
    overall_mean = _overall_mean(result.product_summary)
    comparison = _comparison_summary(result.monthly_aggregates)

    charts = [
        _trend_chart(result.monthly_aggregates),
        _bar_chart(result.quality.by_rejection_reason, "rejection_reason", "rejected_row_count", "Причины отклонения"),
        _bar_chart(result.quality.by_product, "product", "rejection_rate", "Доля отклонений по продуктам", percent=True),
    ]
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Панель UMUX-Lite</title><style>{_STYLE}</style>
<script>{get_plotlyjs()}</script></head><body>
<main><h1>Панель UMUX-Lite</h1><p class="lede">Самодостаточная сводка по агрегатам обработанных данных. Рейтинги показывают относительные результаты; универсального порога «хорошего UMUX» не предполагается.</p>
<section class="cards" aria-label="Ключевые показатели">
{_card("Исходные ответы", str(raw))}{_card("Принятые ответы", str(accepted))}{_card("Доля отклонений", _percent(rejection_rate))}{_card("Средний UMUX", _number(overall_mean))}
</section>
{_interactive_response_list(result.accepted)}
<section><h2>Сравнение продуктов и версий</h2>{_comparison_table(comparison)}</section>
<section><h2>Тренды UMUX по месяцам</h2><p>Подпись каждой точки показывает число принятых ответов. Пропуски означают календарные месяцы без принятых ответов для этой пары продукта и версии и не считаются изменением.</p>{charts[0]}</section>
<section><h2>Наибольшие отрицательные изменения по месяцам</h2>{_negative_changes(result.monthly_aggregates)}</section>
<section><h2>Причины отклонения</h2>{charts[1]}{_insight_table(result.quality.by_rejection_reason, ["rejection_reason", "rejected_row_count"], ["Причина", "Отклонено строк"])}</section>
<section><h2>Качество данных</h2>{charts[2]}{_insight_table(result.quality.by_product, ["product", "raw_row_count", "accepted_row_count", "rejected_row_count", "rejection_rate"], ["Продукт", "Исходные", "Принятые", "Отклонённые", "Доля отклонений"], percent_columns={"rejection_rate"})}</section>
</main></body></html>"""


def _overall_mean(product_summary: pd.DataFrame) -> float | None:
    columns = {"total_valid_responses", "overall_mean_umux"}
    if product_summary.empty or not columns.issubset(product_summary.columns):
        return None
    counts = pd.to_numeric(product_summary["total_valid_responses"], errors="coerce").fillna(0)
    means = pd.to_numeric(product_summary["overall_mean_umux"], errors="coerce")
    total = counts.sum()
    return float((counts * means).sum() / total) if total else None


def _interactive_response_list(accepted: pd.DataFrame) -> str:
    dimensions = ["product", "product_version", "platform", "country", "user_segment"]
    if accepted.empty or not set(dimensions).issubset(accepted.columns):
        return '<section><h2>Интерактивный список ответов</h2><p class="empty">Нет принятых ответов для фильтрации.</p></section>'

    columns = [column for column in ["response_id", "submitted_at", *dimensions, "umux_score"] if column in accepted.columns]
    records = accepted.loc[:, columns].where(pd.notna(accepted.loc[:, columns]), None).to_dict("records")
    payload = _safe_json(records)
    labels = {
        "response_id": "Идентификатор ответа",
        "submitted_at": "Дата ответа",
        "product": "Продукт",
        "product_version": "Версия",
        "platform": "Платформа",
        "country": "Страна",
        "user_segment": "Сегмент пользователя",
        "umux_score": "UMUX",
    }
    filters = "".join(
        f'<label>{labels[field]}<select id="filter-{field}" data-filter="{field}"><option value="">Все значения</option></select></label>'
        for field in dimensions
    )
    header = "".join(f"<th>{html.escape(labels.get(column, column))}</th>" for column in columns)
    return f'''<section><h2>Интерактивный список ответов</h2>
<p>Выберите одно или несколько значений. Список и доступные значения остальных фильтров будут показывать только подходящие ответы.</p>
<div class="filters">{filters}</div><p id="filtered-response-count" aria-live="polite"></p>
<div class="table-wrap filtered-response-table"><table><thead><tr>{header}</tr></thead><tbody id="filtered-responses"></tbody></table></div>
<script>
const filterableRecords = {payload};
const filterFields = ["product", "product_version", "platform", "country", "user_segment"];
const responseColumns = {_safe_json(columns)};
const responseBody = document.getElementById("filtered-responses");
const responseCount = document.getElementById("filtered-response-count");
function applyFilters() {{
  const selected = Object.fromEntries(filterFields.map((field) => [field, document.getElementById(`filter-${{field}}`).value]));
  const matching = filterableRecords.filter((record) => filterFields.every((field) => !selected[field] || record[field] === selected[field]));
  filterFields.forEach((field) => {{
    const availableValues = new Set(filterableRecords
      .filter((record) => filterFields.every((other) => other === field || !selected[other] || record[other] === selected[other]))
      .map((record) => String(record[field])));
    const select = document.getElementById(`filter-${{field}}`);
    Array.from(select.options).forEach((option) => {{
      if (!option.value) return;
      option.disabled = !availableValues.has(option.value);
      option.hidden = option.disabled;
    }});
  }});
  responseBody.replaceChildren(...matching.map((record) => {{
    const row = document.createElement("tr");
    responseColumns.forEach((column) => {{
      const cell = document.createElement("td");
      cell.textContent = record[column] ?? "—";
      row.append(cell);
    }});
    return row;
  }}));
  responseCount.textContent = `Подходящих ответов: ${{matching.length}}`;
}}
filterFields.forEach((field) => {{
  const select = document.getElementById(`filter-${{field}}`);
  [...new Set(filterableRecords.map((record) => String(record[field])))].sort((left, right) => left.localeCompare(right, "ru")).forEach((value) => {{
    const option = new Option(value, value);
    select.add(option);
  }});
  select.addEventListener("change", applyFilters);
}});
applyFilters();
</script></section>'''


def _safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _comparison_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"product", "product_version", "valid_responses", "mean_umux"}
    if monthly.empty or not required.issubset(monthly.columns):
        return pd.DataFrame(columns=["product", "product_version", "total_valid_responses", "overall_mean_umux"])
    rows: list[dict[str, object]] = []
    for (product, version), group in monthly.groupby(["product", "product_version"], sort=True, dropna=False):
        counts = pd.to_numeric(group["valid_responses"], errors="coerce").fillna(0)
        means = pd.to_numeric(group["mean_umux"], errors="coerce")
        total = counts.sum()
        if total:
            rows.append(
                {
                    "product": product,
                    "product_version": version,
                    "total_valid_responses": int(total),
                    "overall_mean_umux": float((counts * means).sum() / total),
                }
            )
    return pd.DataFrame(rows)


def _trend_chart(monthly: pd.DataFrame) -> str:
    required = {"month", "product", "product_version", "valid_responses", "mean_umux"}
    if monthly.empty or not required.issubset(monthly.columns):
        return '<p class="empty">Данные о трендах по месяцам отсутствуют.</p>'
    figure = go.Figure()
    for (product, version), group in monthly.groupby(["product", "product_version"], sort=True, dropna=False):
        by_month = group.sort_values("month", kind="stable").set_index(pd.to_datetime(group.sort_values("month", kind="stable")["month"]))
        calendar = pd.date_range(by_month.index.min(), by_month.index.max(), freq="MS")
        values = by_month.reindex(calendar)
        figure.add_trace(go.Scatter(x=calendar, y=values["mean_umux"], mode="lines+markers+text", name=f"{product} {version}", text=[f"Ответов: {int(v)}" if pd.notna(v) else "" for v in values["valid_responses"]], textposition="top center", hovertemplate="%{x|%m.%Y}<br>UMUX: %{y:.1f}<br>%{text}<extra></extra>", connectgaps=False))
    figure.update_layout(yaxis=dict(title="Средний UMUX", range=[0, 100]), xaxis=dict(title="Календарный месяц", tickformat="%m.%Y"), legend_title="Продукт / версия", margin=dict(l=45, r=20, t=30, b=45))
    return _figure_html(figure, "monthly-trends")


def _bar_chart(data: pd.DataFrame, label: str, value: str, title: str, *, percent: bool = False) -> str:
    if data.empty or label not in data or value not in data:
        return '<p class="empty">Данные отсутствуют.</p>'
    values = pd.to_numeric(data[value], errors="coerce").fillna(0)
    figure = go.Figure(go.Bar(x=data[label].astype(str), y=values, text=[_percent(v) if percent else str(int(v)) for v in values], textposition="auto"))
    figure.update_layout(title=title, yaxis_title="Доля" if percent else "Строки", margin=dict(l=45, r=20, t=45, b=75))
    return _figure_html(figure, title.lower().replace(" ", "-"))


def _figure_html(figure: go.Figure, identifier: str) -> str:
    payload = json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    safe_id = html.escape(identifier, quote=True)
    return f'<div id="{safe_id}" class="chart"></div><script>Plotly.newPlot("{safe_id}", {payload}, {{responsive:true}});</script>'


def _comparison_table(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return '<p class="empty">Нет принятых ответов для сравнения продуктов и версий.</p>'
    return _insight_table(comparison.sort_values(["product", "product_version"], kind="stable"), ["product", "product_version", "total_valid_responses", "overall_mean_umux"], ["Продукт", "Версия", "Ответы", "СРЕДНИЙ UMUX"])


def _negative_changes(monthly: pd.DataFrame) -> str:
    required = {"month", "product", "product_version", "month_over_month_delta", "valid_responses"}
    if monthly.empty or not required.issubset(monthly.columns):
        return '<p class="empty">Нет данных об изменениях по календарным месяцам.</p>'
    negative = monthly.loc[pd.to_numeric(monthly["month_over_month_delta"], errors="coerce") < 0].copy()
    if negative.empty:
        return '<p class="empty">Отрицательных изменений по календарным месяцам нет.</p>'
    return _insight_table(negative.sort_values("month_over_month_delta", kind="stable").head(10), ["product", "product_version", "month", "month_over_month_delta", "valid_responses"], ["Продукт", "Версия", "Месяц", "Изменение", "Ответы"])


def _insight_table(data: pd.DataFrame, columns: list[str], headings: list[str], *, percent_columns: set[str] | None = None) -> str:
    if data.empty:
        return '<p class="empty">Данные отсутствуют.</p>'
    percent_columns = percent_columns or set()
    header = "".join(f"<th>{html.escape(title)}</th>" for title in headings)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(_format(column, row.get(column), column in percent_columns))}</td>" for column in columns) + "</tr>" for _, row in data.iterrows())
    return f'<div class="table-wrap"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def _format(column: str, value: object, percent: bool = False) -> str:
    if pd.isna(value):
        return "—"
    if percent:
        return _percent(float(value))
    if column in {"month", "latest_month"}:
        return _month(value)
    if column in {"mean_umux", "latest_mean_umux", "overall_mean_umux", "month_over_month_delta"}:
        return _number(float(value))
    if column.endswith("responses") or column.endswith("row_count"):
        return str(int(value))
    return _display(value)


def _card(label: str, value: str) -> str:
    return f'<article class="card"><h2>{html.escape(label)}</h2><p>{html.escape(value)}</p></article>'


def _number(value: float | None) -> str:
    return "—" if value is None or pd.isna(value) else f"{value:.1f}"


def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _month(value: object) -> str:
    timestamp = pd.Timestamp(value)
    months = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря")
    return f"{months[timestamp.month - 1]} {timestamp.year}"


def _display(value: object) -> str:
    return str(value)


_STYLE = """
body{margin:0;background:#f5f7fb;color:#172033;font:16px system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}main{max-width:1200px;margin:auto;padding:28px}h1{margin-bottom:0}.lede{color:#4d5b73}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:24px 0}.card,section{background:#fff;border:1px solid #dfe5ef;border-radius:10px;padding:18px;margin:20px 0;box-shadow:0 1px 2px #1720330d}.card h2{font-size:.9rem;margin:0;color:#4d5b73}.card p{font-size:2rem;font-weight:700;margin:8px 0 0}.filters{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px}.filters label{display:grid;gap:4px;color:#4d5b73;font-weight:600}.filters select{border:1px solid #bac5d8;border-radius:5px;background:#fff;padding:8px;color:#172033;font:inherit}.chart{min-height:380px}.table-wrap{overflow-x:auto}.filtered-response-table{max-height:420px;overflow:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #e4e8f0}th{background:#f8faff}.empty{color:#4d5b73;font-style:italic}code{background:#f1f3f7;padding:2px 4px;border-radius:3px}@media(max-width:600px){main{padding:16px}.chart{min-height:320px}}
"""
