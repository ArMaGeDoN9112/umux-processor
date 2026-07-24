from __future__ import annotations

from itertools import product
from pathlib import Path

import pandas as pd
import pytest

from umux_processor.cleaning import clean_records, deduplicate_records
from umux_processor.config import load_configuration
from umux_processor.scoring import UMUXScoreError, calculate_umux_score, score_accepted_records


CONFIG = load_configuration(Path(__file__).parents[1] / "config" / "normalization.toml")


def records(rows: list[dict[str, object]]) -> pd.DataFrame:
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
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


@pytest.mark.parametrize(
    ("score1", "score2", "expected"),
    [(1, 1, 0.0), (3, 3, 50.0), (5, 5, 100.0), (1, 5, 50.0), (5, 1, 50.0)],
)
def test_calculates_required_boundary_and_mixed_scores(score1: int, score2: int, expected: float) -> None:
    assert calculate_umux_score(score1, score2) == expected


@pytest.mark.parametrize("score1, score2", list(product(range(1, 6), repeat=2)))
def test_calculates_every_valid_score_combination(score1: int, score2: int) -> None:
    expected = ((score1 - 1) + (score2 - 1)) / 8 * 100

    assert calculate_umux_score(score1, score2) == expected


@pytest.mark.parametrize("score1, score2", [(0, 1), (1, 6), (3.5, 3), (True, 3), ("3", 3)])
def test_rejects_internal_values_outside_validated_integer_domain(score1: object, score2: object) -> None:
    with pytest.raises(UMUXScoreError):
        calculate_umux_score(score1, score2)


def test_scores_only_deduplicated_accepted_rows_and_preserves_rejected_rows() -> None:
    cleaned = clean_records(
        records(
            [
                {"response_id": "accepted", "score1": "3", "score2": "5", "source_row": 2},
                {"response_id": "rejected", "score1": "bad", "score2": "5", "source_row": 3},
                {"response_id": "duplicate", "score1": "4", "score2": "2", "source_row": 4},
                {"response_id": "duplicate", "score1": "4", "score2": "2", "source_row": 5},
            ]
        ),
        CONFIG,
    )
    deduplicated = deduplicate_records(cleaned)
    rejected_before_scoring = deduplicated.rejected.copy(deep=True)

    scored = score_accepted_records(deduplicated.accepted)

    assert scored["response_id"].tolist() == ["accepted", "duplicate"]
    assert scored["umux_score"].tolist() == [75.0, 50.0]
    assert "umux_score" not in rejected_before_scoring.columns
    pd.testing.assert_frame_equal(deduplicated.rejected, rejected_before_scoring)


def test_dataframe_scoring_preserves_input_order_and_row_lineage() -> None:
    accepted = pd.DataFrame(
        [
            {"response_id": "second", "score1": 5, "score2": 1, "source_file": "/input/b.csv", "source_row": 9},
            {"response_id": "first", "score1": 1, "score2": 2, "source_file": "/input/a.csv", "source_row": 2},
        ]
    )

    scored = score_accepted_records(accepted)

    assert scored["response_id"].tolist() == accepted["response_id"].tolist()
    assert scored[["source_file", "source_row"]].to_dict("records") == accepted[
        ["source_file", "source_row"]
    ].to_dict("records")
    assert scored["umux_score"].between(0, 100, inclusive="both").all()
