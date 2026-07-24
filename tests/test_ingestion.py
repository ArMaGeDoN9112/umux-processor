from __future__ import annotations

from pathlib import Path

import pytest

from umux_processor.ingestion import IngestionError, ingest_csv_inputs


HEADER = (
    "response_id,submitted_at,product,product_version,platform,country,"
    "user_segment,score1,score2"
)
ROW = "response-1,2024-01-02 03:04:05,Payments,2.0,Web,US,New,5,4"


def write_csv(path: Path, rows: list[str], header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_ingests_one_valid_file_with_string_values_and_lineage(tmp_path: Path) -> None:
    source = write_csv(
        tmp_path / "responses.csv",
        [ROW, "response-2,2024-01-03 03:04:05,Search,02.00,,,,,"],
    )

    result = ingest_csv_inputs([source])

    assert result.input_files == (source.resolve(),)
    assert result.records["product_version"].tolist() == ["2.0", "02.00"]
    assert result.records["platform"].tolist() == ["Web", ""]
    assert result.records["source_file"].tolist() == [str(source.resolve())] * 2
    assert result.records["source_row"].tolist() == [2, 3]


def test_combines_multiple_explicit_files_in_argument_order(tmp_path: Path) -> None:
    second = write_csv(tmp_path / "second.csv", [ROW.replace("response-1", "second")])
    first = write_csv(tmp_path / "first.csv", [ROW.replace("response-1", "first")])

    result = ingest_csv_inputs([second, first])

    assert result.records["response_id"].tolist() == ["second", "first"]
    assert result.records["source_input_order"].tolist() == [0, 1]


def test_expands_globs_in_sorted_order_and_removes_duplicate_matches(tmp_path: Path) -> None:
    write_csv(tmp_path / "b.csv", [ROW.replace("response-1", "b")])
    first = write_csv(tmp_path / "a.csv", [ROW.replace("response-1", "a")])

    result = ingest_csv_inputs([tmp_path / "*.csv", first])

    assert [path.name for path in result.input_files] == ["a.csv", "b.csv"]
    assert result.records["response_id"].tolist() == ["a", "b"]


@pytest.mark.parametrize("input_path", ["missing.csv", "no-match-*.csv"])
def test_reports_missing_file_or_unmatched_pattern(tmp_path: Path, input_path: str) -> None:
    with pytest.raises(IngestionError, match="did not match any files"):
        ingest_csv_inputs([tmp_path / input_path])


def test_reports_missing_required_columns_at_file_level(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "missing-column.csv", ["one,2024-01-02 03:04:05"], "response_id,submitted_at")

    with pytest.raises(IngestionError, match=r"missing required columns: .*product"):
        ingest_csv_inputs([source])


def test_retains_extra_columns_and_logs_a_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    source = write_csv(tmp_path / "extra.csv", [f"{ROW},campaign"], f"{HEADER},marketing_campaign")

    result = ingest_csv_inputs([source])

    assert result.records["marketing_campaign"].tolist() == ["campaign"]
    assert "outside the processing schema" in caplog.text


def test_reports_empty_and_malformed_csv_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    malformed = tmp_path / "malformed.csv"
    malformed.write_text(f"{HEADER}\n{ROW},unexpected\n", encoding="utf-8")

    with pytest.raises(IngestionError, match="empty CSV with no usable schema"):
        ingest_csv_inputs([empty])
    with pytest.raises(IngestionError, match="malformed CSV"):
        ingest_csv_inputs([malformed])


def test_prevents_generated_output_artifacts_from_becoming_inputs(tmp_path: Path) -> None:
    generated = write_csv(tmp_path / "cleaned_responses.csv", [ROW])

    with pytest.raises(IngestionError, match="generated output artifact"):
        ingest_csv_inputs([generated])
