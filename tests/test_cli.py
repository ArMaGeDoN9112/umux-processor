from __future__ import annotations

import logging
import os
from pathlib import Path
import subprocess
import sys


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


def test_module_startup_loads_the_default_configuration() -> None:
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

    assert result.returncode == 0


def test_import_does_not_configure_root_logging() -> None:
    root_logger = logging.getLogger()
    original_handlers = list(root_logger.handlers)

    __import__("umux_processor.logging")

    assert root_logger.handlers == original_handlers
