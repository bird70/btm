"""CLI: btm-slope — compute slope in degrees (Horn 1981)."""

from __future__ import annotations

import argparse
import sys

from btm._logging import configure_cli_logging, get_logger
from btm.core.slope import compute_slope
from btm.io.raster import RasterDataset

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-slope", description="Compute slope in degrees."
    )
    p.add_argument(
        "--bathy", required=True, metavar="PATH", help="Input bathymetric raster"
    )
    p.add_argument(
        "--output", required=True, metavar="PATH", help="Output GeoTIFF path"
    )
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
    except Exception as exc:  # noqa: BLE001
        _log.exception("Unexpected error: %s", exc)
        return 5
    result = compute_slope(ds.array, ds.cell_size(), nodata=ds.nodata)
    ds.array = result
    ds.to_file(args.output, dtype="float32")
    _log.info("Saved slope raster to %s", args.output)
    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
