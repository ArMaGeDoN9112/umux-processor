from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from umux_processor.cleaning import clean_records, deduplicate_records
from umux_processor.config import load_configuration


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
    ("product", "expected"),
    [
        (" payment ", "Payments"),
        ("payments", "Payments"),
        ("CHECK OUT", "Checkout"),
        ("checkout", "Checkout"),
        ("Onboarding", "Onboarding"),
        (" onboardng ", "Onboarding"),
        (" PROFILE ", "Profile"),
        ("search", "Search"),
        (" SERCH ", "Search"),
    ],
)
def test_normalizes_all_configured_product_aliases(product: str, expected: str) -> None:
    result = clean_records(records([{ "product": product }]), CONFIG)

    assert result.currently_valid["product"].tolist() == [expected]


@pytest.mark.parametrize(
    ("platform", "country", "user_segment", "expected"),
    [
        (" web ", " us ", " new ", ("Web", "US", "New")),
        ("ANDROID", "gb", "EXISTING", ("Android", "GB", "Existing")),
        ("iOs", "de", "returning", ("iOS", "DE", "Returning")),
    ],
)
def test_normalizes_casing_and_whitespace(
    platform: str, country: str, user_segment: str, expected: tuple[str, str, str]
) -> None:
    result = clean_records(
        records([{ "platform": platform, "country": country, "user_segment": user_segment }]), CONFIG
    )

    assert tuple(result.currently_valid.loc[0, ["platform", "country", "user_segment"]]) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [("platform", ""), ("country", "  "), ("user_segment", None)],
)
def test_missing_optional_fields_become_unknown(field: str, value: object) -> None:
    result = clean_records(records([{field: value}]), CONFIG)

    assert result.currently_valid.loc[0, field] == "Unknown"


@pytest.mark.parametrize(
    ("field", "value"),
    [("platform", "Desktop"), ("country", "USA"), ("country", "1!"), ("user_segment", "VIP")],
)
def test_invalid_nonblank_optional_categories_become_unknown(field: str, value: str) -> None:
    result = clean_records(records([{field: value}]), CONFIG)

    assert result.currently_valid.loc[0, field] == "Unknown"


@pytest.mark.parametrize("submitted_at", ["", "not a date", "2024/01/02 03:04:05"])
def test_rejects_missing_or_malformed_dates(submitted_at: str) -> None:
    result = clean_records(records([{ "submitted_at": submitted_at }]), CONFIG)

    assert result.currently_valid.empty
    assert result.rejected.loc[0, "rejection_reasons"] == ("invalid_submitted_at",)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("score1", "", "missing_score1"),
        ("score2", None, "missing_score2"),
        ("score1", "three", "non_integer_score1"),
        ("score2", "4.5", "non_integer_score2"),
        ("score1", "0", "score1_out_of_range"),
        ("score2", "6.0", "score2_out_of_range"),
    ],
)
def test_rejects_invalid_scores(field: str, value: object, reason: str) -> None:
    result = clean_records(records([{field: value}]), CONFIG)

    assert result.currently_valid.empty
    assert result.rejected.loc[0, "rejection_reasons"] == (reason,)


@pytest.mark.parametrize(("value", "expected"), [("3", 3), ("3.0", 3), ("3.00", 3), (3.0, 3)])
def test_accepts_exactly_integral_score_representations(value: object, expected: int) -> None:
    result = clean_records(records([{ "score1": value }]), CONFIG)

    assert result.currently_valid.loc[0, "score1"] == expected


def test_collects_all_rejection_reasons_for_a_row() -> None:
    result = clean_records(
        records(
            [
                {
                    "response_id": " ",
                    "submitted_at": "bad",
                    "product": "Unmapped product",
                    "product_version": "",
                    "score1": "1.5",
                    "score2": "9",
                }
            ]
        ),
        CONFIG,
    )

    assert result.currently_valid.empty
    assert result.rejected.loc[0, "rejection_reasons"] == (
        "missing_response_id",
        "invalid_submitted_at",
        "unknown_product",
        "missing_product_version",
        "non_integer_score1",
        "score2_out_of_range",
    )


def test_preserves_original_values_extra_columns_and_lineage() -> None:
    input_records = records(
        [{"product": " payment ", "score1": "3.0", "source_file": "/input/a.csv", "source_row": 17, "campaign": "A"}]
    )
    result = clean_records(input_records, CONFIG)

    valid = result.currently_valid.loc[0]
    assert valid["original_product"] == " payment "
    assert valid["original_score1"] == "3.0"
    assert valid["campaign"] == "A"
    assert valid["source_file"] == "/input/a.csv"
    assert valid["source_row"] == 17
    assert valid["product"] == "Payments"


def test_rejected_rows_preserve_original_values_and_lineage() -> None:
    result = clean_records(
        records(
            [{"submitted_at": "bad", "product": " payment ", "source_file": "/input/b.csv", "source_row": 23}]
        ),
        CONFIG,
    )

    rejected = result.rejected.loc[0]
    assert rejected["original_submitted_at"] == "bad"
    assert rejected["original_product"] == " payment "
    assert rejected["source_file"] == "/input/b.csv"
    assert rejected["source_row"] == 23


def test_preserves_a_nonblank_product_version_as_a_trimmed_string() -> None:
    result = clean_records(records([{ "product_version": " 02.00 " }]), CONFIG)

    assert result.currently_valid.loc[0, "product_version"] == "02.00"
    assert result.currently_valid.loc[0, "original_product_version"] == " 02.00 "


def test_every_input_row_is_classified_once() -> None:
    input_records = records([{},{"score1": "bad"},{"product": "Unknown"}])
    result = clean_records(input_records, CONFIG)

    assert len(result.currently_valid) + len(result.rejected) == len(input_records)


def test_deduplicates_equivalent_valid_copies_by_normalized_questionnaire_payload() -> None:
    cleaned = clean_records(
        records(
            [
                {"response_id": " answer-1 ", "product": " payment ", "platform": " web ", "score1": "3.0"},
                {"response_id": "answer-1", "product": "Payments", "platform": "Web", "score1": "3"},
                {"response_id": "answer-1", "product": "Payments", "platform": "Web", "score1": "3"},
            ]
        ),
        CONFIG,
    )

    result = deduplicate_records(cleaned)

    assert len(result.accepted) == 1
    assert len(result.rejected) == 2
    assert result.rejected["rejection_reasons"].tolist() == [("duplicate_exact",), ("duplicate_exact",)]
    assert result.rejected["duplicate_context"].tolist() == ["duplicate_exact", "duplicate_exact"]


def test_keeps_one_valid_copy_and_retains_invalid_copy_reasons_with_context() -> None:
    cleaned = clean_records(records([{}, {"score1": "bad"}]), CONFIG)

    result = deduplicate_records(cleaned)

    assert len(result.accepted) == 1
    assert result.rejected.loc[0, "rejection_reasons"] == ("non_integer_score1",)
    assert result.rejected.loc[0, "duplicate_context"] == "duplicate_with_valid"


def test_rejects_all_conflicting_valid_copies_without_selecting_a_winner() -> None:
    cleaned = clean_records(records([{ "score1": "3" }, {"score1": "5"}]), CONFIG)

    result = deduplicate_records(cleaned)

    assert result.accepted.empty
    assert result.rejected["rejection_reasons"].tolist() == [
        ("duplicate_conflict",),
        ("duplicate_conflict",),
    ]
    assert result.rejected["duplicate_context"].tolist() == ["duplicate_conflict", "duplicate_conflict"]


def test_keeps_all_invalid_duplicate_copies_with_original_reasons_and_context() -> None:
    cleaned = clean_records(records([{ "score1": "bad" }, {"score2": "9"}]), CONFIG)

    result = deduplicate_records(cleaned)

    assert result.accepted.empty
    assert result.rejected["rejection_reasons"].tolist() == [
        ("non_integer_score1",),
        ("score2_out_of_range",),
    ]
    assert result.rejected["duplicate_context"].tolist() == ["duplicate_all_invalid", "duplicate_all_invalid"]


def test_deduplicates_ids_across_files_and_uses_lineage_order_when_dataframe_is_reordered() -> None:
    input_records = records(
        [
            {"source_file": "/input/b.csv", "source_row": 2, "source_input_order": 0},
            {"source_file": "/input/a.csv", "source_row": 9, "source_input_order": 1},
        ]
    )
    cleaned = clean_records(input_records, CONFIG)
    reordered = clean_records(input_records.iloc[::-1].reset_index(drop=True), CONFIG)

    result = deduplicate_records(cleaned)
    reordered_result = deduplicate_records(reordered)

    assert result.accepted[["source_file", "source_row"]].to_dict("records") == [
        {"source_file": "/input/b.csv", "source_row": 2}
    ]
    assert reordered_result.accepted[["source_file", "source_row"]].to_dict("records") == [
        {"source_file": "/input/b.csv", "source_row": 2}
    ]
    assert result.input_count == len(input_records)
    assert len(result.accepted) + len(result.rejected) == len(input_records)
