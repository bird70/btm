"""CLI: btm-scale-compare — multi-scale neighbourhood comparison."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage

from btm._logging import configure_cli_logging, get_logger
from btm.io.raster import RasterDataset

_log = get_logger(__name__)

_SUPPORTED_FILTERS = frozenset(["median", "uniform", "gaussian", "minimum", "maximum"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-scale-compare",
        description="Compare a raster across a range of neighbourhood sizes.",
    )
    p.add_argument("--bathy", required=True, metavar="PATH")
    p.add_argument(
        "--filter", default="median", choices=sorted(_SUPPORTED_FILTERS), metavar="FILTER"
    )
    p.add_argument("--min-nbhs", required=True, type=int, metavar="INT")
    p.add_argument("--max-nbhs", required=True, type=int, metavar="INT")
    p.add_argument("--outdir", required=True, metavar="PATH")
    p.add_argument("--num-sizes", type=int, default=10, metavar="INT",
                   help="Number of neighbourhood sizes to sample (default 10)")
    _add_log_args(p)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_cli_logging(verbose=args.verbose, quiet=args.quiet, log_file=args.log_file)

    try:
        ds = RasterDataset.from_file(args.bathy)
    except FileNotFoundError as exc:
        _log.error("%s", exc)
        return 2

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arr = ds.array.astype(np.float64)

    sizes = np.linspace(args.min_nbhs, args.max_nbhs, num=args.num_sizes,
                        endpoint=True).astype(np.uint32)

    filter_func = getattr(scipy.ndimage, f"{args.filter}_filter", None)
    if filter_func is None:
        _log.error("Unknown filter: %s", args.filter)
        return 1

    stem = Path(args.bathy).stem
    for size in sizes:
        filtered = filter_func(arr, size=int(size))
        out_path = str(outdir / f"{stem}_{args.filter}_{size:03d}.tif")
        tmp = RasterDataset(
            path=None, array=filtered.astype(np.float32), crs=ds.crs,
            transform=ds.transform, nodata=ds.nodata,
            cell_width=ds.cell_width, cell_height=ds.cell_height,
        )
        tmp.to_file(out_path, dtype="float32")
        print(out_path)

    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
