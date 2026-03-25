"""CLI: btm-depth-statistics — focal depth statistics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from btm._logging import configure_cli_logging, get_logger
from btm.core.depth_statistics import compute_focal_stats
from btm.io.raster import RasterDataset

_log = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-depth-stats",
        description="Compute focal depth statistics over a bathymetric raster.",
    )
    p.add_argument("--bathy", required=True, metavar="PATH")
    p.add_argument("--neighborhood", required=True, type=int, metavar="INT")
    p.add_argument("--outdir", required=True, metavar="PATH")
    p.add_argument("--stats", required=True, nargs="+", metavar="STAT",
                   help="Stats: mean std variance iqr kurtosis")
    p.add_argument("--window", default="rectangle", choices=["rectangle", "circle"])
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

    try:
        results = compute_focal_stats(
            ds.array, args.neighborhood, args.stats,
            window_type=args.window, nodata=ds.nodata,
        )
    except ValueError as exc:
        _log.error("Parameter error: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        _log.exception("Unexpected error: %s", exc)
        return 5

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.bathy).stem
    n_label = f"{args.neighborhood:03d}"

    for stat, arr in results.items():
        out_path = str(outdir / f"{stem}_{stat}_{n_label}.tif")
        tmp_ds = RasterDataset(
            path=None, array=arr.astype("float32"), crs=ds.crs,
            transform=ds.transform, nodata=None,
            cell_width=ds.cell_width, cell_height=ds.cell_height,
        )
        tmp_ds.to_file(out_path, dtype="float32")
        print(out_path)

    return 0


def _add_log_args(p: argparse.ArgumentParser) -> None:
    g = p.add_mutually_exclusive_group()
    g.add_argument("--verbose", action="store_true")
    g.add_argument("--quiet", action="store_true")
    p.add_argument("--log-file", metavar="PATH", default=None)


if __name__ == "__main__":
    sys.exit(main())
