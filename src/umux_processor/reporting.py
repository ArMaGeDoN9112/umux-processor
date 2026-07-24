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
            handle.write("<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>Dashboard unavailable</title><body><h1>Dashboard unavailable</h1><p>Dashboard generation failed. The CSV and JSON audit artifacts were generated successfully; consult the pipeline logs for details.</p></body></html>")
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
    current = _current_combinations(result.monthly_aggregates)
    small_samples = _small_samples(result.monthly_aggregates, threshold)

    charts = [
        _trend_chart(result.monthly_aggregates),
        _bar_chart(result.quality.by_rejection_reason, "rejection_reason", "rejected_row_count", "Rejection reasons"),
        _bar_chart(result.quality.by_product, "product", "rejection_rate", "Rejection rate by product", percent=True),
    ]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>UMUX-Lite dashboard</title><style>{_STYLE}</style>
<script>{get_plotlyjs()}</script></head><body>
<main><h1>UMUX-Lite dashboard</h1><p class="lede">A standalone summary of the tested pipeline aggregates. Rankings show relative performance; no universal “good UMUX” threshold is assumed.</p>
<section class="cards" aria-label="Key performance indicators">
{_card("Raw responses", str(raw))}{_card("Accepted responses", str(accepted))}{_card("Rejection rate", _percent(rejection_rate))}{_card("Overall mean UMUX", _number(overall_mean))}
</section>
<section><h2>Product and version comparison</h2>{_comparison_table(current)}</section>
<section><h2>Monthly UMUX trends</h2><p>Every marker label is its accepted response count. Gaps represent calendar months with no accepted response for that product/version; they are not treated as a change.</p>{charts[0]}</section>
<section><h2>Lowest-performing latest combinations</h2>{_insight_table(current.sort_values(["latest_mean_umux", "product", "product_version"], kind="stable") if not current.empty else current, ["product", "product_version", "latest_month", "latest_mean_umux", "total_valid_responses"], ["Product", "Version", "Latest month", "Latest UMUX", "Responses"])}</section>
<section><h2>Largest negative calendar-month changes</h2>{_negative_changes(result.monthly_aggregates)}</section>
<section><h2>Rejection reasons</h2>{charts[1]}{_insight_table(result.quality.by_rejection_reason, ["rejection_reason", "rejected_row_count"], ["Reason", "Rejected rows"])}</section>
<section><h2>Data quality</h2>{charts[2]}{_insight_table(result.quality.by_product, ["product", "raw_row_count", "accepted_row_count", "rejected_row_count", "rejection_rate"], ["Product", "Raw", "Accepted", "Rejected", "Rejection rate"], percent_columns={"rejection_rate"})}</section>
{_small_sample_notice(small_samples, threshold)}
<section><h2>How to read these metrics</h2><p>UMUX-Lite is calculated for accepted questionnaires only: <code>((score1 − 1) + (score2 − 1)) / 8 × 100</code>. The overall mean is the response-count-weighted combination of the provided product summaries. Rejection rate is rejected raw rows divided by all ingested raw rows, including excluded duplicate copies.</p></section>
</main></body></html>"""


def _overall_mean(product_summary: pd.DataFrame) -> float | None:
    columns = {"total_valid_responses", "overall_mean_umux"}
    if product_summary.empty or not columns.issubset(product_summary.columns):
        return None
    counts = pd.to_numeric(product_summary["total_valid_responses"], errors="coerce").fillna(0)
    means = pd.to_numeric(product_summary["overall_mean_umux"], errors="coerce")
    total = counts.sum()
    return float((counts * means).sum() / total) if total else None


def _current_combinations(monthly: pd.DataFrame) -> pd.DataFrame:
    required = {"month", "product", "product_version", "valid_responses", "mean_umux"}
    if monthly.empty or not required.issubset(monthly.columns):
        return pd.DataFrame(columns=["product", "product_version", "total_valid_responses", "latest_month", "latest_mean_umux"])
    rows: list[dict[str, object]] = []
    for (product, version), group in monthly.groupby(["product", "product_version"], sort=True, dropna=False):
        ordered = group.sort_values("month", kind="stable")
        latest = ordered.iloc[-1]
        rows.append({"product": product, "product_version": version, "total_valid_responses": int(ordered["valid_responses"].sum()), "latest_month": latest["month"], "latest_mean_umux": float(latest["mean_umux"])})
    return pd.DataFrame(rows)


def _small_samples(monthly: pd.DataFrame, threshold: int) -> pd.DataFrame:
    required = {"month", "product", "product_version", "valid_responses"}
    if monthly.empty or not required.issubset(monthly.columns):
        return pd.DataFrame(columns=["month", "product", "product_version", "valid_responses"])
    return monthly.loc[monthly["valid_responses"] < threshold, ["month", "product", "product_version", "valid_responses"]].sort_values(["month", "product", "product_version"], kind="stable")


def _trend_chart(monthly: pd.DataFrame) -> str:
    required = {"month", "product", "product_version", "valid_responses", "mean_umux"}
    if monthly.empty or not required.issubset(monthly.columns):
        return '<p class="empty">No monthly trend data is available.</p>'
    figure = go.Figure()
    for (product, version), group in monthly.groupby(["product", "product_version"], sort=True, dropna=False):
        by_month = group.sort_values("month", kind="stable").set_index(pd.to_datetime(group.sort_values("month", kind="stable")["month"]))
        calendar = pd.date_range(by_month.index.min(), by_month.index.max(), freq="MS")
        values = by_month.reindex(calendar)
        figure.add_trace(go.Scatter(x=calendar, y=values["mean_umux"], mode="lines+markers+text", name=f"{product} {version}", text=[f"n={int(v)}" if pd.notna(v) else "" for v in values["valid_responses"]], textposition="top center", hovertemplate="%{x|%b %Y}<br>UMUX: %{y:.1f}<br>%{text}<extra></extra>", connectgaps=False))
    figure.update_layout(yaxis=dict(title="Mean UMUX", range=[0, 100]), xaxis_title="Calendar month", legend_title="Product / version", margin=dict(l=45, r=20, t=30, b=45))
    return _figure_html(figure, "monthly-trends")


def _bar_chart(data: pd.DataFrame, label: str, value: str, title: str, *, percent: bool = False) -> str:
    if data.empty or label not in data or value not in data:
        return '<p class="empty">No data is available.</p>'
    values = pd.to_numeric(data[value], errors="coerce").fillna(0)
    figure = go.Figure(go.Bar(x=data[label].astype(str), y=values, text=[_percent(v) if percent else str(int(v)) for v in values], textposition="auto"))
    figure.update_layout(title=title, yaxis_title="Rate" if percent else "Rows", margin=dict(l=45, r=20, t=45, b=75))
    return _figure_html(figure, title.lower().replace(" ", "-"))


def _figure_html(figure: go.Figure, identifier: str) -> str:
    payload = json.dumps(figure.to_plotly_json(), cls=PlotlyJSONEncoder, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    safe_id = html.escape(identifier, quote=True)
    return f'<div id="{safe_id}" class="chart"></div><script>Plotly.newPlot("{safe_id}", {payload}, {{responsive:true}});</script>'


def _comparison_table(current: pd.DataFrame) -> str:
    if current.empty:
        return '<p class="empty">No accepted responses are available for product/version comparison.</p>'
    return _insight_table(current.sort_values(["product", "product_version"], kind="stable"), ["product", "product_version", "total_valid_responses", "latest_month", "latest_mean_umux"], ["Product", "Version", "Responses", "Latest month", "Latest UMUX"])


def _negative_changes(monthly: pd.DataFrame) -> str:
    required = {"month", "product", "product_version", "month_over_month_delta", "valid_responses"}
    if monthly.empty or not required.issubset(monthly.columns):
        return '<p class="empty">No calendar-month changes are available.</p>'
    negative = monthly.loc[pd.to_numeric(monthly["month_over_month_delta"], errors="coerce") < 0].copy()
    if negative.empty:
        return '<p class="empty">No negative calendar-month changes are available.</p>'
    return _insight_table(negative.sort_values("month_over_month_delta", kind="stable").head(10), ["product", "product_version", "month", "month_over_month_delta", "valid_responses"], ["Product", "Version", "Month", "Change", "Responses"])


def _small_sample_notice(samples: pd.DataFrame, threshold: int) -> str:
    if samples.empty:
        return f'<section class="notice"><h2>Small-sample notice</h2><p>No product/version/month groups are below the configured threshold of {threshold} responses.</p></section>'
    items = "".join(f"<li>{html.escape(_display(row.product))} {html.escape(_display(row.product_version))}, {html.escape(_month(row.month))}: n={int(row.valid_responses)}</li>" for row in samples.itertuples(index=False))
    return f'<section class="notice"><h2>Small-sample notice</h2><p>The following product/version/month groups are below the configured threshold of {threshold} responses; treat rankings and changes as directional.</p><ul>{items}</ul></section>'


def _insight_table(data: pd.DataFrame, columns: list[str], headings: list[str], *, percent_columns: set[str] | None = None) -> str:
    if data.empty:
        return '<p class="empty">No data is available.</p>'
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
    if column in {"mean_umux", "latest_mean_umux", "month_over_month_delta"}:
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
    return pd.Timestamp(value).strftime("%b %Y")


def _display(value: object) -> str:
    return str(value)


_STYLE = """
body{margin:0;background:#f5f7fb;color:#172033;font:16px system-ui,-apple-system,Segoe UI,sans-serif;line-height:1.45}main{max-width:1200px;margin:auto;padding:28px}h1{margin-bottom:0}.lede{color:#4d5b73}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:16px;margin:24px 0}.card,section{background:#fff;border:1px solid #dfe5ef;border-radius:10px;padding:18px;margin:20px 0;box-shadow:0 1px 2px #1720330d}.card h2{font-size:.9rem;margin:0;color:#4d5b73}.card p{font-size:2rem;font-weight:700;margin:8px 0 0}.chart{min-height:380px}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid #e4e8f0}th{background:#f8faff}.notice{border-left:5px solid #e59b24}.empty{color:#4d5b73;font-style:italic}code{background:#f1f3f7;padding:2px 4px;border-radius:3px}@media(max-width:600px){main{padding:16px}.chart{min-height:320px}}
"""
