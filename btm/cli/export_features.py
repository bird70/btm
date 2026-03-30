"""CLI: btm-export-features — extract BTM terrain derivatives at sample points.

Outputs a CSV containing the original point columns plus one column per BTM
derivative, suitable for merging into an ML training pipeline
(e.g. benthic_model).

Usage examples
--------------
Basic extraction::

    btm-export-features \\
      --bathy   data/bathymetry.tif \\
      --points  data/train.csv \\
      --output  data/train_btm.csv

With optional rule-based class code and derivative rasters::

    btm-export-features \\
      --bathy      data/bathymetry.tif \\
      --points     data/train.csv \\
      --classdict  data/classification.csv \\
      --include-rule-class \\
      --outdir     data/btm_rasters \\
      --output     data/train_btm.csv

Exit codes
----------
0  Success
1  Invalid parameters
2  Input file not found
3  CSV format error
5  Unexpected error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from btm._logging import configure_cli_logging, get_logger

_log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="btm-export-features",
        description=(
            "Extract BTM terrain derivatives at sample-point locations "
            "and write a CSV for use in ML pipelines (e.g. benthic_model)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--bathy",
        required=True,
        metavar="PATH",
        help="Input single-band bathymetric GeoTIFF.",
    )
    p.add_argument(
        "--points",
        required=True,
        metavar="CSV",
        help=(
            "Input CSV with columns ID, x, y (coordinates must match "
            "the raster CRS).  Additional columns are preserved."
        ),
    )
    p.add_argument(
        "--output",
        required=True,
        metavar="CSV",
        help="Output CSV path (original columns + btm_* columns).",
    )

    # BPI radii
    bpi = p.add_argument_group("BPI radii (in cells)")
    bpi.add_argument(
        "--broad-inner",
        type=int,
        default=10,
        metavar="INT",
        help="Broad BPI annulus inner radius.",
    )
    bpi.add_argument(
        "--broad-outer",
        type=int,
        default=30,
        metavar="INT",
        help="Broad BPI annulus outer radius.",
    )
    bpi.add_argument(
        "--fine-inner",
        type=int,
        default=1,
        metavar="INT",
        help="Fine BPI annulus inner radius.",
    )
    bpi.add_argument(
        "--fine-outer",
        type=int,
        default=5,
        metavar="INT",
        help="Fine BPI annulus outer radius.",
    )

    # Optional extras
    p.add_argument(
        "--include-rule-class",
        action="store_true",
        help=(
            "Add a btm_rule_class column containing the integer class code "
            "from the BTM CON cascade.  Requires --classdict."
        ),
    )
    p.add_argument(
        "--classdict",
        default=None,
        metavar="PATH",
        help="Classification dictionary (.csv, .xml, .xlsx) — required with "
        "--include-rule-class.",
    )
    p.add_argument(
        "--no-interactions",
        action="store_true",
        help="Skip derived interaction features (bpi_magnitude, etc.).",
    )
    p.add_argument(
        "--include-eco-features",
        action="store_true",
        help=(
            "Compute and append northness, eastness, max curvature, and "
            "complexity eco-derivatives to the output CSV."
        ),
    )
    p.add_argument(
        "--prefix",
        default="btm_",
        metavar="STR",
        help="Column name prefix for all BTM-derived columns.",
    )
    p.add_argument(
        "--outdir",
        default=None,
        metavar="PATH",
        help=(
            "If given, write all intermediate derivative rasters as "
            "LZW-compressed GeoTIFFs to this directory."
        ),
    )
    p.add_argument(
        "--x-col",
        default="x",
        metavar="COL",
        help="Name of the x-coordinate column in --points CSV.",
    )
    p.add_argument(
        "--y-col",
        default="y",
        metavar="COL",
        help="Name of the y-coordinate column in --points CSV.",
    )
    p.add_argument(
        "--id-col",
        default="ID",
        metavar="COL",
        help="Name of the identifier column in --points CSV.",
    )

    # Logging
    log = p.add_argument_group("logging")
    log.add_argument("--verbose", "-v", action="store_true")
    log.add_argument("--quiet", "-q", action="store_true")
    log.add_argument("--log-file", metavar="PATH", default=None)

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0911
    args = build_parser().parse_args(argv)
    configure_cli_logging(
        verbose=args.verbose, quiet=args.quiet, log_file=args.log_file
    )

    # ── validate inputs ─────────────────────────────────────────────────────
    bathy_path = Path(args.bathy)
    if not bathy_path.exists():
        _log.error("Bathymetry raster not found: %s", bathy_path)
        return 2

    points_path = Path(args.points)
    if not points_path.exists():
        _log.error("Points CSV not found: %s", points_path)
        return 2

    if args.include_rule_class and args.classdict is None:
        _log.error("--include-rule-class requires --classdict to be specified")
        return 1

    if args.classdict is not None and not Path(args.classdict).exists():
        _log.error("Classification file not found: %s", args.classdict)
        return 2

    # ── pandas / btm.features are optional deps; give a helpful message ──────
    try:
        import pandas as pd
    except ImportError:
        _log.error(
            "pandas is required for btm-export-features.  "
            "Install it with:  pip install 'btm[ml]'"
        )
        return 5

    # ── load points CSV ──────────────────────────────────────────────────────
    try:
        points = pd.read_csv(points_path)
    except Exception as exc:
        _log.error("Could not read points CSV: %s", exc)
        return 3

    for col in (args.id_col, args.x_col, args.y_col):
        if col not in points.columns:
            _log.error(
                "Required column %r not found in %s  (columns: %s)",
                col,
                points_path.name,
                list(points.columns),
            )
            return 3

    # Rename user-supplied column names to the canonical ID/x/y if different
    rename = {}
    if args.id_col != "ID":
        rename[args.id_col] = "ID"
    if args.x_col != "x":
        rename[args.x_col] = "x"
    if args.y_col != "y":
        rename[args.y_col] = "y"
    if rename:
        points = points.rename(columns=rename)

    _log.info("Loaded %d sample points from %s", len(points), points_path.name)

    # ── extract BTM features ─────────────────────────────────────────────────
    try:
        from btm.features.extract import extract_btm_features

        result = extract_btm_features(
            sample_points=points,
            bathymetry_tif=str(bathy_path),
            broad_inner=args.broad_inner,
            broad_outer=args.broad_outer,
            fine_inner=args.fine_inner,
            fine_outer=args.fine_outer,
            include_rule_class=args.include_rule_class,
            classification_file=args.classdict,
            include_interactions=not args.no_interactions,
            include_eco_features=args.include_eco_features,
            btm_prefix=args.prefix,
            outdir=args.outdir,
        )
    except (TypeError, ValueError) as exc:
        _log.error("Feature extraction failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        _log.exception("Unexpected error during feature extraction: %s", exc)
        return 5

    # Reverse column renames so the output preserves original column names
    if rename:
        result = result.rename(columns={v: k for k, v in rename.items()})

    # ── write output CSV ─────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        result.to_csv(out_path, index=False)
    except OSError as exc:
        _log.error("Could not write output CSV: %s", exc)
        return 5

    btm_cols = [c for c in result.columns if c.startswith(args.prefix)]
    _log.info(
        "Wrote %d rows × %d BTM columns to %s",
        len(result),
        len(btm_cols),
        out_path,
    )
    print(f"Output: {out_path.resolve()}")
    print(f"BTM columns ({len(btm_cols)}): {', '.join(btm_cols)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
