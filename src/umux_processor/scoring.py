"""UMUX-Lite score calculation for deduplicated accepted questionnaires."""

from __future__ import annotations

from numbers import Integral, Real

import pandas as pd


class UMUXScoreError(ValueError):
    """Raised when scoring receives values outside the validated score domain."""


def calculate_umux_score(score1: object, score2: object) -> float:
    """Return the UMUX-Lite score for two validated integer responses.

    This is intentionally strict: validation owns conversion from raw inputs,
    while scoring requires the canonical 1--5 integer domain.
    """
    _validate_score(score1, "score1")
    _validate_score(score2, "score2")
    return float(((score1 - 1) + (score2 - 1)) / 8 * 100)  # type: ignore[operator]


def score_accepted_records(accepted: pd.DataFrame) -> pd.DataFrame:
    """Attach scores to already validated and deduplicated accepted records.

    The input is not mutated. Row order and all existing columns, including
    source lineage, are retained; rejected records are intentionally not an
    input to this function.
    """
    missing_columns = {"score1", "score2"}.difference(accepted.columns)
    if missing_columns:
        raise UMUXScoreError(f"Accepted records are missing score columns: {', '.join(sorted(missing_columns))}")

    scored = accepted.copy(deep=True)
    scored["umux_score"] = pd.Series(
        [calculate_umux_score(score1, score2) for score1, score2 in accepted[["score1", "score2"]].itertuples(index=False, name=None)],
        index=scored.index,
        dtype="float64",
    )
    return scored


def _validate_score(value: object, field: str) -> None:
    is_integral_real = isinstance(value, Real) and value.is_integer() if not isinstance(value, Integral) else True
    if isinstance(value, bool) or not is_integral_real or not 1 <= value <= 5:
        raise UMUXScoreError(f"{field} must be a validated integer from 1 through 5; got {value!r}")
