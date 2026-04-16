"""Shared logging configuration for ARCO tool scripts."""

from __future__ import annotations

import logging
import os

from pyparsing import Optional


def configure_logging(level: Optional[str] = None) -> None:
    """Configure the root logger with a standard format for ARCO tools.

    Sets up the root logger to emit messages to the console (stderr) with a
    consistent format.  Call this function once inside the
    ``if __name__ == "__main__":`` block of each tool script before calling
    ``main()``.

    Args:
        level: Minimum log level to emit.  Defaults to ``logging.INFO``.
            Pass ``logging.DEBUG`` to see detailed algorithm entry/exit
            messages from the ``arco`` package.
    """
    if level is None:
        level = os.getenv("ARCO_LOG_LEVEL", "INFO").upper()
    else:
        level = level.upper()

    print("SETTING UP LEVEL", level)

    logging.basicConfig(
        level=level,
        format="%(levelname)s - %(message)s",
    )
