"""CLI: btm-run-model — run the complete BTM pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from btm._logging import configure_cli_logging, get_logger
from btm.core.model import run_full_model

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-run-model",
        description="Run the complete BTM pipeline.",
    )
    p.add_argument("--bathy", required=True, metavar="PATH")
    p.add_argument("--broad-inner", required=True, type=int, metavar="INT")
    p.add_argument("--broad-outer", required=True, type=int, metavar="INT")
    p.add_argument("--fine-inner", required=True, type=int, metavar="INT")
    p.add_argument("--fine-outer", required=True, type=int, metavar="INT")
    p.add_argument("--classdict", required=True, metavar="PATH")
    p.add_argument("--outdir", required=True, metavar="PATH")
    p.add_argument(
        "--no-intermediates",
        action="store_true",
        help="Write only classified_zones.tif",
    )
    p.add_argument("--block-size", type=int, default=None, metavar="INT")
    _add_log_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(
        verbose=args.verbose, quiet=args.quiet, log_file=args.log_file
    )

    # Validate inputs before starting
    for attr, label in [("bathy", "--bathy"), ("classdict", "--classdict")]:
        path = getattr(args, attr)
        if not Path(path).exists():
            _log.error("File not found for %s: %s", label, path)
            return 2

    try:
        outputs = run_full_model(
            bathy_path=args.bathy,
            broad_bpi_inner=args.broad_inner,
            broad_bpi_outer=args.broad_outer,
            fine_bpi_inner=args.fine_inner,
            fine_bpi_outer=args.fine_outer,
            classification_file=args.classdict,
            outdir=args.outdir,
            keep_intermediates=not args.no_intermediates,
            block_size=args.block_size,
        )
    except FileNotFoundError as exc:
        _log.error("File not found: %s", exc)
        return 2
    except (TypeError, ValueError) as exc:
        if "classification" in str(exc).lower() or "classdict" in str(exc).lower():
            _log.error("Classification dictionary error: %s", exc)
            return 3
        _log.error("Invalid parameters: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        _log.exception("Unexpected error: %s", exc)
        return 5

    import numpy as np
    import rasterio

    classified_path = outputs.get("classified_zones")
    if classified_path:
        with rasterio.open(classified_path) as src:
            arr = src.read(1)
        if not np.any(arr > 0):
            _log.warning("No cells were classified.")
            return 4

    for name, path in outputs.items():
        print(path)

    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
