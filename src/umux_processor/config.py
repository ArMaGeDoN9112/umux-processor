"""Typed loading and validation for normalization configuration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tomllib
from typing import Any, Mapping


class ConfigurationError(ValueError):
    """Raised when a normalization configuration cannot be used safely."""


@dataclass(frozen=True)
class CountryPolicy:
    case: str
    missing_value: str
    code_length: int


@dataclass(frozen=True)
class ReportSettings:
    small_sample_threshold: int


@dataclass(frozen=True)
class NormalizationConfig:
    product_aliases: Mapping[str, str]
    supported_platforms: tuple[str, ...]
    user_segment_aliases: Mapping[str, str]
    countries: CountryPolicy
    timestamp_format: str
    report: ReportSettings


def load_configuration(path: str | Path) -> NormalizationConfig:
    """Load a TOML configuration file and return its validated typed values."""
    config_path = Path(path)
    try:
        with config_path.open("rb") as config_file:
            contents = tomllib.load(config_file)
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigurationError(f"Invalid TOML in {config_path}: {error}") from error
    except OSError as error:
        raise ConfigurationError(f"Could not read configuration file {config_path}: {error}") from error

    return _parse_configuration(contents)


def _parse_configuration(contents: Mapping[str, Any]) -> NormalizationConfig:
    products = _required_section(contents, "products")
    platforms = _required_section(contents, "platforms")
    user_segments = _required_section(contents, "user_segments")
    countries = _required_section(contents, "countries")
    timestamps = _required_section(contents, "timestamps")
    report = _required_section(contents, "report")

    return NormalizationConfig(
        product_aliases=_aliases(products, "products.aliases"),
        supported_platforms=_string_list(platforms.get("supported"), "platforms.supported"),
        user_segment_aliases=_aliases(user_segments, "user_segments.aliases"),
        countries=CountryPolicy(
            case=_choice(countries.get("case"), "countries.case", {"upper"}),
            missing_value=_non_empty_string(countries.get("missing_value"), "countries.missing_value"),
            code_length=_positive_integer(countries.get("code_length"), "countries.code_length"),
        ),
        timestamp_format=_timestamp_format(timestamps.get("accepted_format")),
        report=ReportSettings(
            small_sample_threshold=_positive_integer(
                report.get("small_sample_threshold"), "report.small_sample_threshold"
            )
        ),
    )


def _required_section(contents: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = contents.get(name)
    if not isinstance(section, Mapping):
        raise ConfigurationError(f"Missing required section: [{name}]")
    return section


def _aliases(section: Mapping[str, Any], field: str) -> Mapping[str, str]:
    aliases = section.get("aliases")
    if not isinstance(aliases, Mapping):
        raise ConfigurationError(f"{field} must be a table")
    if not aliases:
        raise ConfigurationError(f"{field} must not be empty")
    validated_aliases: dict[str, str] = {}
    normalized_aliases: set[str] = set()
    for alias, canonical in aliases.items():
        validated_alias = _non_empty_string(alias, f"{field} key")
        normalized_alias = validated_alias.strip().casefold()
        if normalized_alias in normalized_aliases:
            raise ConfigurationError(f"{field} contains duplicate normalized alias: {normalized_alias}")
        normalized_aliases.add(normalized_alias)
        validated_aliases[validated_alias] = _non_empty_string(canonical, f"{field}.{alias}")
    return validated_aliases


def _string_list(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ConfigurationError(f"{field} must be a non-empty list of strings")
    values = tuple(_non_empty_string(item, field) for item in value)
    if len(set(values)) != len(values):
        raise ConfigurationError(f"{field} must not contain duplicate values")
    return values


def _choice(value: Any, field: str, allowed: set[str]) -> str:
    value = _non_empty_string(value, field)
    if value not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ConfigurationError(f"{field} must be one of: {choices}")
    return value


def _positive_integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError(f"{field} must be a positive integer")
    return value


def _timestamp_format(value: Any) -> str:
    field = "timestamps.accepted_format"
    timestamp_format = _non_empty_string(value, field)
    if "%" not in timestamp_format:
        raise ConfigurationError(f"{field} must contain a strptime directive")
    try:
        datetime.strptime("2000-01-02 03:04:05", timestamp_format)
    except ValueError as error:
        raise ConfigurationError(f"{field} is not a valid strptime format: {error}") from error
    return timestamp_format


def _non_empty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value
