"""
Demo script: run the full BTM pipeline on the Fagatele Bay test data and
print a human-readable summary of the outputs.

Usage
-----
    python scripts/run_fagatelebay.py [--outdir PATH] [--classdict PATH]

If --outdir already contains outputs from a previous run they are re-used
(the pipeline is NOT re-run).  Use --force to always recompute.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# Ensure the repo root is on the path so that `btm` is importable even
# without a `pip install -e .`.
# ---------------------------------------------------------------------------
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import rasterio  # noqa: E402

from btm.core.model import run_full_model  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DATA = _REPO_ROOT / "tests" / "data"
_DEFAULT_BATHY = _DATA / "bathy5m_clip.tif"
_DEFAULT_CLASSDICT = _DATA / "fagatelebay.csv"
_DEFAULT_OUTDIR = _REPO_ROOT / "outputs" / "fagatelebay"

_CLASS_NAMES = {
    1: "Reef Crest",
    2: "Mid-Slope Ridges",
    3: "Back Reef",
    4: "Upper Slopes",
    5: "Lower Bank Shelf",
    6: "Upper Reef Flat",
    7: "Open Slopes",
    8: "Depression",
    9: "Back Reef",
    10: "Mid-Slope Depressions",
    11: "Deep Depression",
}


def _raster_stats(path: str) -> dict:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True)
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "nodata_count": int(arr.mask.sum()) if hasattr(arr, "mask") else 0,
    }


def _class_distribution(path: str) -> list[tuple[int, int]]:
    with rasterio.open(path) as src:
        arr = src.read(1)
    codes, counts = np.unique(arr, return_counts=True)
    return list(zip(codes.tolist(), counts.tolist()))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bathy",
        default=str(_DEFAULT_BATHY),
        metavar="PATH",
        help=f"Input bathymetry raster (default: {_DEFAULT_BATHY.name})",
    )
    p.add_argument(
        "--classdict",
        default=str(_DEFAULT_CLASSDICT),
        metavar="PATH",
        help=f"Classification dictionary (default: {_DEFAULT_CLASSDICT.name})",
    )
    p.add_argument(
        "--outdir",
        default=str(_DEFAULT_OUTDIR),
        metavar="PATH",
        help=f"Output directory (default: {_DEFAULT_OUTDIR})",
    )
    p.add_argument(
        "--broad-inner",
        type=int,
        default=10,
        metavar="N",
        help="Broad BPI inner radius in cells (default: 10)",
    )
    p.add_argument(
        "--broad-outer",
        type=int,
        default=30,
        metavar="N",
        help="Broad BPI outer radius in cells (default: 30)",
    )
    p.add_argument(
        "--fine-inner",
        type=int,
        default=1,
        metavar="N",
        help="Fine BPI inner radius in cells (default: 1)",
    )
    p.add_argument(
        "--fine-outer",
        type=int,
        default=5,
        metavar="N",
        help="Fine BPI outer radius in cells (default: 5)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if output directory already exists",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    outdir = pathlib.Path(args.outdir)
    classified_path = outdir / "classified_zones.tif"

    # ------------------------------------------------------------------
    # Run the model (or skip if outputs already exist)
    # ------------------------------------------------------------------
    if classified_path.exists() and not args.force:
        print(f"Outputs already exist in {outdir}  (use --force to recompute)\n")
        outputs = {
            "broad_bpi": str(outdir / "broad_bpi.tif"),
            "fine_bpi": str(outdir / "fine_bpi.tif"),
            "broad_std": str(outdir / "broad_std.tif"),
            "fine_std": str(outdir / "fine_std.tif"),
            "slope": str(outdir / "slope.tif"),
            "classified_zones": str(classified_path),
        }
    else:
        print("Running BTM pipeline …")
        bathy = pathlib.Path(args.bathy)
        if not bathy.exists():
            sys.exit(f"ERROR: bathy raster not found: {bathy}")
        cd = pathlib.Path(args.classdict)
        if not cd.exists():
            sys.exit(f"ERROR: classification file not found: {cd}")

        outputs = run_full_model(
            bathy_path=str(bathy),
            broad_bpi_inner=args.broad_inner,
            broad_bpi_outer=args.broad_outer,
            fine_bpi_inner=args.fine_inner,
            fine_bpi_outer=args.fine_outer,
            classification_file=str(cd),
            outdir=str(outdir),
        )
        print("Done.\n")

    # ------------------------------------------------------------------
    # Print raster statistics
    # ------------------------------------------------------------------
    print("=== BTM output summary ===\n")
    fmt = "{:<16} min={:>8.1f}  max={:>8.1f}  mean={:>8.2f}  nodata={}"
    for name, path in outputs.items():
        if name == "classified_zones":
            continue
        if not pathlib.Path(path).exists():
            print(f"  {name}: output file missing")
            continue
        s = _raster_stats(path)
        print(fmt.format(name, s["min"], s["max"], s["mean"], s["nodata_count"]))

    # ------------------------------------------------------------------
    # Print class distribution
    # ------------------------------------------------------------------
    print("\n=== Classified zones distribution ===\n")
    total_pixels = None
    dist = _class_distribution(outputs["classified_zones"])
    total_pixels = sum(c for _, c in dist)

    header = f"{'Code':>4}  {'Zone':<24}  {'Pixels':>8}  {'%':>6}"
    print(header)
    print("-" * len(header))
    for code, count in sorted(dist):
        pct = count / total_pixels * 100
        label = _CLASS_NAMES.get(
            code, "(unclassified)" if code == 0 else f"class {code}"
        )
        print(f"{code:>4}  {label:<24}  {count:>8}  {pct:>5.1f}%")

    print(f"\nTotal pixels: {total_pixels:,}")
    print(f"\nOutput directory: {outdir.resolve()}")


if __name__ == "__main__":
    main()
