"""CLI: btm-classify — classify benthic terrain from pre-computed derivatives."""

from __future__ import annotations

import argparse
import sys

from btm._logging import configure_cli_logging, get_logger
from btm.classification.reader import read_classification
from btm.core.classify import classify_terrain
from btm.io.raster import RasterDataset

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="btm-classify", description="Classify benthic terrain.")
    p.add_argument("--broad-std", required=True, metavar="PATH")
    p.add_argument("--fine-std", required=True, metavar="PATH")
    p.add_argument("--slope", required=True, metavar="PATH")
    p.add_argument("--bathy", required=True, metavar="PATH")
    p.add_argument("--classdict", required=True, metavar="PATH")
    p.add_argument("--output", required=True, metavar="PATH")
    _add_log_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)

    try:
        broad_ds = RasterDataset.from_file(args.broad_std)
        fine_ds = RasterDataset.from_file(args.fine_std)
        slope_ds = RasterDataset.from_file(args.slope)
        bathy_ds = RasterDataset.from_file(args.bathy)
    except FileNotFoundError as exc:
        _log.error("Input file not found: %s", exc)
        return 2

    try:
        classes = read_classification(args.classdict)
    except (TypeError, ValueError) as exc:
        _log.error("Classification file error: %s", exc)
        return 3

    import numpy as np
    result = classify_terrain(
        broad_ds.array.astype(np.float64),
        fine_ds.array.astype(np.float64),
        slope_ds.array.astype(np.float64),
        bathy_ds.array.astype(np.float64),
        classes,
        nodata=bathy_ds.nodata,
    )

    import numpy as np
    if not np.any(result > 0):
        _log.warning("No cells were classified (all unclassified).")
        return 4

    bathy_ds.array = result
    bathy_ds.nodata = 0
    bathy_ds.to_file(args.output, dtype="int32")
    _log.info("Saved classified raster to %s", args.output)
    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
