"""Shared logging helpers for the BTM package and its CLI entry points."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with a NullHandler (library convention)."""
    logger = logging.getLogger(name)
    logger.addHandler(logging.NullHandler())
    return logger


def configure_cli_logging(
    verbose: bool = False,
    quiet: bool = False,
    log_file: str | None = None,
) -> None:
    """Configure root logger for CLI usage.

    Parameters
    ----------
    verbose:
        Set log level to DEBUG.
    quiet:
        Set log level to WARNING (overridden by *verbose*).
    log_file:
        If provided, also write log records to this file path.
    """
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    else:
        level = logging.INFO

    root = logging.getLogger()
    root.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(formatter)
        root.addHandler(fh)
