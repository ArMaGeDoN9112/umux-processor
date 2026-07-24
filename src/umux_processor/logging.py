"""Application logging helpers.

Importing this module deliberately does not alter global logging configuration.
"""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure application logging when an executable entry point requests it."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
