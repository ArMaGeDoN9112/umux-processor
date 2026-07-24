from __future__ import annotations

from pathlib import Path

import pytest

from umux_processor.config import ConfigurationError, load_configuration


DEFAULT_CONFIG = Path(__file__).parents[1] / "config" / "normalization.toml"


def test_loads_default_configuration() -> None:
    config = load_configuration(DEFAULT_CONFIG)

    assert config.product_aliases["payment"] == "Payments"
    assert config.supported_platforms == ("Web", "Android", "iOS")
    assert config.timestamp_format == "%Y-%m-%d %H:%M:%S"
    assert config.report.small_sample_threshold == 30


def test_reports_missing_required_section(tmp_path: Path) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text("[products]\naliases = { payment = 'Payments' }\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match=r"Missing required section: \[platforms\]"):
        load_configuration(config_path)


@pytest.mark.parametrize(
    ("alias_definition", "message"),
    [
        ("aliases = { payment = '' }", "products.aliases.payment must be a non-empty string"),
        ("aliases = ['Payment']", "products.aliases must be a table"),
    ],
)
def test_reports_invalid_product_alias_definitions(
    tmp_path: Path, alias_definition: str, message: str
) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text(
        "\n".join(
            [
                "[products]",
                alias_definition,
                "[platforms]",
                "supported = ['Web']",
                "[user_segments]",
                "aliases = { new = 'New' }",
                "[countries]",
                "case = 'upper'",
                "missing_value = 'Unknown'",
                "code_length = 2",
                "[timestamps]",
                "accepted_format = '%Y-%m-%d'",
                "[report]",
                "small_sample_threshold = 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match=message):
        load_configuration(config_path)


@pytest.mark.parametrize("threshold", ["0", "-1", "1.5", "true"])
def test_reports_invalid_small_sample_threshold(tmp_path: Path, threshold: str) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            "small_sample_threshold = 30", f"small_sample_threshold = {threshold}"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="report.small_sample_threshold must be a positive integer"):
        load_configuration(config_path)


def test_reports_malformed_toml(tmp_path: Path) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text("[products\naliases = {}", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid TOML"):
        load_configuration(config_path)


def test_reports_ambiguous_product_aliases(tmp_path: Path) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'payments = "Payments"', 'payments = "Payments"\n" payment " = "Payments"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="duplicate normalized alias: payment"):
        load_configuration(config_path)


def test_reports_invalid_timestamp_format(tmp_path: Path) -> None:
    config_path = tmp_path / "normalization.toml"
    config_path.write_text(
        DEFAULT_CONFIG.read_text(encoding="utf-8").replace(
            'accepted_format = "%Y-%m-%d %H:%M:%S"', 'accepted_format = "not-a-format"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ConfigurationError, match="timestamps.accepted_format must contain a strptime directive"):
        load_configuration(config_path)
