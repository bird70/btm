"""CLI: btm-vrm — compute Vector Ruggedness Measure."""

from __future__ import annotations

import argparse
import sys

from btm._logging import configure_cli_logging, get_logger
from btm.core.vrm import compute_vrm
from btm.io.raster import RasterDataset

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-vrm",
        description="Compute Vector Ruggedness Measure (VRM).",
    )
    p.add_argument("--bathy", required=True, metavar="PATH")
    p.add_argument("--neighborhood-size", required=True, type=int, metavar="INT")
    p.add_argument("--output", required=True, metavar="PATH")
    _add_log_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(
        verbose=args.verbose, quiet=args.quiet, log_file=args.log_file
    )
    try:
        ds = RasterDataset.from_file(args.bathy)
    except FileNotFoundError as exc:
        _log.error("%s", exc)
        return 2
    try:
        result = compute_vrm(
            ds.array, args.neighborhood_size, ds.cell_size(), nodata=ds.nodata
        )
    except ValueError as exc:
        _log.error("Invalid parameters: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        _log.exception("Unexpected error: %s", exc)
        return 5
    ds.array = result
    ds.to_file(args.output, dtype="float32")
    _log.info("Saved VRM raster to %s", args.output)
    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
