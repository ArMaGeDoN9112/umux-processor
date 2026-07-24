from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from umux_processor.artifacts import (
    ARTIFACT_FILENAMES,
    CLEANED_RESPONSE_COLUMNS,
    PRODUCT_SUMMARY_ARTIFACT_COLUMNS,
    REJECTED_RESPONSE_COLUMNS,
    ArtifactWriteError,
    write_audit_artifacts,
)
from umux_processor.config import load_configuration
from umux_processor.pipeline import run_pipeline


CONFIG = load_configuration(Path(__file__).parents[1] / "config" / "normalization.toml")


def _records(rows: list[dict[str, object]]) -> pd.DataFrame:
    defaults = {
        "response_id": "response-1",
        "submitted_at": "2024-01-02 03:04:05",
        "product": "Payments",
        "product_version": "2.0",
        "platform": "Web",
        "country": "US",
        "user_segment": "New",
        "score1": "5",
        "score2": "4",
        "source_file": "/input/responses.csv",
        "source_row": 2,
        "source_input_order": 0,
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


def _pipeline_result() -> object:
    return run_pipeline(
        _records(
            [
                {"response_id": "accepted", "score1": "3", "score2": "5", "source_row": 2},
                {"response_id": "invalid", "score1": "bad", "score2": "5", "source_row": 3},
                {"response_id": "duplicate", "score1": "4", "score2": "2", "source_row": 4},
                {"response_id": "duplicate", "score1": "4", "score2": "2", "source_row": 5},
            ]
        ),
        CONFIG,
    )


def test_writes_exact_artifact_names_schemas_and_reconciled_counts(tmp_path: Path) -> None:
    result = _pipeline_result()

    paths = write_audit_artifacts(result, tmp_path)

    assert tuple(paths) == ARTIFACT_FILENAMES
    assert {path.name for path in paths.values()} == set(ARTIFACT_FILENAMES)
    assert list(pd.read_csv(paths["cleaned_responses.csv"]).columns) == CLEANED_RESPONSE_COLUMNS
    assert list(pd.read_csv(paths["rejected_responses.csv"]).columns) == REJECTED_RESPONSE_COLUMNS
    assert list(pd.read_csv(paths["product_summary.csv"]).columns) == PRODUCT_SUMMARY_ARTIFACT_COLUMNS
    assert len(pd.read_csv(paths["cleaned_responses.csv"])) + len(pd.read_csv(paths["rejected_responses.csv"])) == 4

    rejected = pd.read_csv(paths["rejected_responses.csv"])
    assert [json.loads(value) for value in rejected["rejection_reasons"]] == [
        ["non_integer_score1"],
        ["duplicate_exact"],
    ]
    quality = json.loads(paths["quality_summary.json"].read_text(encoding="utf-8"))
    assert quality == {
        "accepted_row_count": 2,
        "duplicate_row_count": 1,
        "raw_row_count": 4,
        "rejected_row_count": 2,
        "rejection_reason_counts": {"duplicate_exact": 1, "non_integer_score1": 1},
    }


def test_repeated_runs_replace_known_artifacts_deterministically_and_preserve_unrelated_files(tmp_path: Path) -> None:
    result = _pipeline_result()
    unrelated = tmp_path / "keep-me.txt"
    unrelated.write_text("unrelated", encoding="utf-8")
    for filename in ARTIFACT_FILENAMES:
        (tmp_path / filename).write_text("obsolete", encoding="utf-8")

    first_paths = write_audit_artifacts(result, tmp_path)
    first_contents = {name: path.read_bytes() for name, path in first_paths.items()}
    second_paths = write_audit_artifacts(result, tmp_path)

    assert {name: path.read_bytes() for name, path in second_paths.items()} == first_contents
    assert unrelated.read_text(encoding="utf-8") == "unrelated"


@pytest.mark.parametrize(
    "rows, expected_cleaned, expected_rejected",
    [
        ([{"score1": "bad"}], 0, 1),
        ([{}], 1, 0),
    ],
)
def test_writes_header_only_csvs_for_empty_dispositions(
    tmp_path: Path, rows: list[dict[str, object]], expected_cleaned: int, expected_rejected: int
) -> None:
    paths = write_audit_artifacts(run_pipeline(_records(rows), CONFIG), tmp_path)

    assert len(pd.read_csv(paths["cleaned_responses.csv"])) == expected_cleaned
    assert len(pd.read_csv(paths["rejected_responses.csv"])) == expected_rejected


def test_serialization_failure_preserves_existing_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = _pipeline_result()
    expected = {filename: f"old-{filename}" for filename in ARTIFACT_FILENAMES}
    for filename, content in expected.items():
        (tmp_path / filename).write_text(content, encoding="utf-8")

    def fail_json(*args: object, **kwargs: object) -> None:
        raise OSError("simulated write failure")

    monkeypatch.setattr("umux_processor.artifacts._write_json_temp", fail_json)

    with pytest.raises(ArtifactWriteError, match="quality_summary.json"):
        write_audit_artifacts(result, tmp_path)

    assert {filename: (tmp_path / filename).read_text(encoding="utf-8") for filename in ARTIFACT_FILENAMES} == expected
