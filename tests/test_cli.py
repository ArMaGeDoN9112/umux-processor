from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys

import pytest

from umux_processor.artifacts import ArtifactWriteError
from umux_processor.cli import EXIT_OUTPUT_ERROR, main
from umux_processor.config import load_configuration
from umux_processor.pipeline import PipelineServiceError, run_pipeline_service


HEADER = (
    "response_id,submitted_at,product,product_version,platform,country,"
    "user_segment,score1,score2"
)


def write_csv(path: Path, rows: list[str], header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def run_cli(command: list[str], *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([*command, *arguments], check=False, capture_output=True, text=True)


def test_module_help_succeeds() -> None:
    project_root = Path(__file__).parents[1]
    environment = os.environ | {"PYTHONPATH": str(project_root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "umux_processor", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0
    assert "UMUX-Lite" in result.stdout


def test_module_requires_input_arguments() -> None:
    project_root = Path(__file__).parents[1]
    environment = os.environ | {"PYTHONPATH": str(project_root / "src")}
    result = subprocess.run(
        [sys.executable, "-m", "umux_processor"],
        check=False,
        capture_output=True,
        text=True,
        cwd=project_root,
        env=environment,
    )

    assert result.returncode == 2
    assert "--input" in result.stderr


def test_module_and_console_entry_points_run_the_same_single_file_pipeline(tmp_path: Path) -> None:
    source = write_csv(
        tmp_path / "responses.csv",
        [
            "accepted,2024-01-02 03:04:05,Payments,2.0,Web,US,New,5,4",
            "rejected,2024-01-03 03:04:05,Payments,2.0,Web,US,New,bad,4",
        ],
    )
    output = tmp_path / "output"
    arguments = ["--input", str(source), "--output", str(output), "--config", "/app/config/normalization.toml"]

    module = run_cli([sys.executable, "-m", "umux_processor"], *arguments)
    console = run_cli(["umux-process"], *arguments)

    assert module.returncode == console.returncode == 0
    assert module.stdout == console.stdout
    assert "raw=2 accepted=1 rejected=1 duplicates=0" in module.stdout
    assert "Pipeline completed:" in module.stdout
    assert "response_id" not in module.stderr
    assert {path.name for path in output.iterdir()} == {
        "cleaned_responses.csv", "rejected_responses.csv", "monthly_aggregates.csv",
        "product_summary.csv", "quality_summary.json", "dashboard.html",
    }


def test_cli_accepts_multiple_files_and_all_rejections_are_a_success(tmp_path: Path) -> None:
    first = write_csv(tmp_path / "first.csv", ["bad-1,not-a-date,Payments,2.0,Web,US,New,bad,4"])
    second = write_csv(tmp_path / "second.csv", ["bad-2,2024-01-03 03:04:05,Payments,2.0,Web,US,New,1,7"])
    output = tmp_path / "output"

    result = run_cli(
        ["umux-process"], "--input", str(first), str(second), "--output", str(output),
        "--config", "/app/config/normalization.toml",
    )

    assert result.returncode == 0
    assert "raw=2 accepted=0 rejected=2 duplicates=0" in result.stdout
    assert (output / "dashboard.html").is_file()
    assert len((output / "cleaned_responses.csv").read_text(encoding="utf-8").splitlines()) == 1


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["--input", "missing.csv", "--output", "/tmp/out", "--config", "/app/config/normalization.toml"], "did not match any files"),
        (["--input", "missing.csv", "--output", "/tmp/out", "--config", "/missing.toml"], "Configuration file not found"),
    ],
)
def test_cli_returns_data_error_for_missing_input_or_invalid_configuration(
    arguments: list[str], expected: str
) -> None:
    result = run_cli(["umux-process"], *arguments)

    assert result.returncode == 3
    assert expected in result.stderr


def test_cli_returns_data_error_for_invalid_schema(tmp_path: Path) -> None:
    source = write_csv(tmp_path / "invalid.csv", ["only,one"], "response_id,submitted_at")

    result = run_cli(
        ["umux-process"], "--input", str(source), "--output", str(tmp_path / "output"),
        "--config", "/app/config/normalization.toml",
    )

    assert result.returncode == 3
    assert "missing required columns" in result.stderr


def test_service_reports_an_unrecoverable_artifact_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = write_csv(tmp_path / "responses.csv", ["ok,2024-01-02 03:04:05,Payments,2.0,Web,US,New,5,4"])

    def fail_artifacts(*args: object, **kwargs: object) -> dict[str, Path]:
        raise ArtifactWriteError("disk unavailable")

    monkeypatch.setattr("umux_processor.artifacts.write_audit_artifacts", fail_artifacts)

    with pytest.raises(PipelineServiceError, match="disk unavailable"):
        run_pipeline_service([source], tmp_path / "output", load_configuration("/app/config/normalization.toml"))

    assert main([
        "--input", str(source), "--output", str(tmp_path / "output"),
        "--config", "/app/config/normalization.toml",
    ]) == EXIT_OUTPUT_ERROR


def test_import_does_not_configure_root_logging() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)

    __import__("umux_processor.logging")

    assert root_logger.handlers == original_handlers


def test_compose_mount_overrides_and_nonroot_runtime_contract_are_packaged_for_review() -> None:
    compose = Path("/app/compose.yaml").read_text(encoding="utf-8")
    dockerfile = Path("/app/Dockerfile").read_text(encoding="utf-8")

    assert "${UMUX_INPUT_DIR:-.}" in compose
    assert "${UMUX_OUTPUT_DIR:-./output}" in compose
    assert "target: /input" in compose and "read_only: true" in compose
    assert "target: /output" in compose
    assert "USER app" in dockerfile
