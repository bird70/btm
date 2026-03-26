"""BTM feature extraction for ML pipelines.

Given a DataFrame of sample points (x, y geographic coordinates) and a
bathymetric raster, this module computes all BTM terrain derivatives on the
full raster and samples their values at each point.

The output DataFrame is intentionally shaped to be *compatible with*
``benthic_model.data.raster_extract.extract_mbes_features``:

- Same index / ``ID`` column
- Additional columns prefixed ``btm_`` (configurable)
- No side-effects: derivative rasters are computed in memory, not written to
  disk unless ``outdir`` is specified

Integration pattern
-------------------
::

    # In benthic_model training pipeline, after extract_mbes_features:
    mbes_df = extract_mbes_features(sample_points, bathy_tif, backscatter_tif)
    btm_df  = extract_btm_features(sample_points, bathy_tif)
    combined = mbes_df.merge(btm_df.drop(columns=["x", "y"]), on="ID")
    engineered = engineer_features(combined, features_config=cfg)

Why this helps
--------------
``benthic_model`` computes TPI (square window) and rugosity, but BTM adds:

- **Standardised BPI** (annular focal mean, z-normalised) at two ecological
  scales: broad (landscape ridges/depressions) and fine (individual biogenic
  structures).  These are the core BTM discriminators the rule-based
  classifier uses, and they expose non-rectangular class boundaries that
  XGBoost/LightGBM can exploit far better than hard thresholds.
- **VRM** — Sappington (2007) Vector Ruggedness Measure.  Theoretically
  more appropriate for seafloor roughness than std-based proxies because it
  captures tri-axial surface orientation variance.
- **Surface-to-planar ratio** — Jenness (2002), sensitive to fine-scale
  complexity at the cell-neighbourhood scale.
- **BTM interaction features** — cross-scale BPI magnitude, VRM × backscatter
  (hard substrate signal), std_btm_context.

The ML model can also receive the BTM rule-based class code as an integer
feature (pass ``include_rule_class=True`` with a ``classification_file``),
which gives the model a strong spatial prior from the expert-defined
classification system.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sample_raster_at_points(
    array: np.ndarray,
    transform,
    xs: np.ndarray,
    ys: np.ndarray,
    nodata: float | None = None,
) -> np.ndarray:
    """Sample a 2-D raster array at given geographic (x, y) coordinates.

    Parameters
    ----------
    array:
        2-D NumPy array (row, col) — the raster band.
    transform:
        Affine transform from rasterio (``rasterio.transform.Affine``).
    xs, ys:
        1-D arrays of x / y coordinates in the same CRS as the raster.
    nodata:
        Value to substitute for out-of-bounds samples (default: NaN).

    Returns
    -------
    np.ndarray
        1-D float array of sampled values, same length as ``xs``.
    """
    import rasterio.transform as rtransform

    rows, cols = rtransform.rowcol(transform, xs, ys)
    rows = np.asarray(rows, dtype=int)
    cols = np.asarray(cols, dtype=int)

    n_rows, n_cols = array.shape
    in_bounds = (rows >= 0) & (rows < n_rows) & (cols >= 0) & (cols < n_cols)

    out = np.full(len(xs), np.nan, dtype=float)
    out[in_bounds] = array[rows[in_bounds], cols[in_bounds]].astype(float)

    # Replace explicit nodata with NaN
    if nodata is not None:
        out[out == nodata] = np.nan

    return out


def extract_btm_features(
    sample_points,  # pd.DataFrame with columns: ID, x, y
    bathymetry_tif: str | Path,
    broad_inner: int = 10,
    broad_outer: int = 30,
    fine_inner: int = 1,
    fine_outer: int = 5,
    include_rule_class: bool = False,
    classification_file: str | Path | None = None,
    include_interactions: bool = True,
    btm_prefix: str = "btm_",
    outdir: str | Path | None = None,
):
    """Compute BTM derivatives and extract values at sample point locations.

    Runs the full BTM feature set (BPI broad/fine, standardised BPI,
    slope, VRM, surface-to-planar ratio) on the input raster, then samples
    each derivative at the supplied (x, y) coordinates.

    Parameters
    ----------
    sample_points:
        ``pandas.DataFrame`` with at minimum columns ``ID``, ``x``, ``y``.
        ``x`` / ``y`` must be in the same CRS as ``bathymetry_tif``.
    bathymetry_tif:
        Path to a single-band bathymetric GeoTIFF.
    broad_inner, broad_outer:
        Inner and outer cell radii for the broad-scale BPI annulus.
    fine_inner, fine_outer:
        Inner and outer cell radii for the fine-scale BPI annulus.
    include_rule_class:
        If ``True``, also sample the BTM rule-based class code (integer 1–N
        as defined in ``classification_file``) as column
        ``btm_rule_class``.  Requires ``classification_file``.
    classification_file:
        Classification dictionary (.csv, .xml, .xlsx) — only required when
        ``include_rule_class=True``.
    include_interactions:
        If ``True``, add derived cross-feature columns (BPI magnitude,
        terrain complexity index, etc.).
    btm_prefix:
        String prefix for all BTM-derived column names.
    outdir:
        Optional directory.  If given, all intermediate rasters are written
        as LZW-compressed GeoTIFFs for inspection / caching.

    Returns
    -------
    pd.DataFrame
        Copy of ``sample_points`` with additional ``btm_*`` columns.
        NaN values indicate out-of-raster or nodata samples.

    Notes
    -----
    Column names produced (default prefix ``btm_``):

    ===============================  =========================================
    ``btm_broad_bpi``                Raw broad-scale BPI (int32)
    ``btm_fine_bpi``                 Raw fine-scale BPI (int32)
    ``btm_broad_std``                Standardised broad BPI (z × 100, int32)
    ``btm_fine_std``                 Standardised fine BPI (z × 100, int32)
    ``btm_slope``                    Slope in degrees (float32)
    ``btm_vrm``                      Vector Ruggedness Measure [0, 1]
    ``btm_surface_ratio``            Surface-to-planar area ratio (≥ 1.0)
    ``btm_rule_class``               Rule-based class code (optional)
    ``btm_bpi_magnitude``            sqrt(broad_std²+fine_std²) — interaction
    ``btm_rough_total``              vrm + surface_ratio − 1 — interaction
    ``btm_broad_x_fine_std``         broad_std × fine_std — interaction
    ===============================  =========================================
    """
    import pandas as pd

    from btm.core.bpi import compute_bpi
    from btm.core.classify import classify_terrain
    from btm.core.slope import compute_slope
    from btm.core.standardize import standardize_bpi
    from btm.core.surface_ratio import compute_surface_planar_ratio
    from btm.core.vrm import compute_vrm
    from btm.io.raster import RasterDataset

    if include_rule_class and classification_file is None:
        raise ValueError(
            "classification_file is required when include_rule_class=True"
        )

    # ------------------------------------------------------------------
    # 1. Load bathymetry
    # ------------------------------------------------------------------
    bathy_ds = RasterDataset.from_file(str(bathymetry_tif))
    bathy = bathy_ds.array
    nodata = bathy_ds.nodata
    cell_size = bathy_ds.cell_size()
    transform = bathy_ds.transform

    _log.info(
        "extract_btm_features: raster %s  shape=%s  cell_size=%.2f",
        Path(bathymetry_tif).name,
        bathy.shape,
        cell_size,
    )

    # ------------------------------------------------------------------
    # 2. Compute BTM derivatives
    # ------------------------------------------------------------------
    broad_bpi = compute_bpi(bathy, broad_inner, broad_outer, cell_size, nodata=nodata)
    fine_bpi = compute_bpi(bathy, fine_inner, fine_outer, cell_size, nodata=nodata)
    broad_std = standardize_bpi(broad_bpi, nodata=0)
    fine_std = standardize_bpi(fine_bpi, nodata=0)
    slope = compute_slope(bathy, cell_size, nodata=nodata)
    vrm = compute_vrm(bathy, neighborhood_size=3, cell_size=cell_size, nodata=nodata)
    surf_ratio = compute_surface_planar_ratio(bathy, cell_size, nodata=nodata)

    derivatives: dict[str, np.ndarray] = {
        "broad_bpi": broad_bpi.astype(float),
        "fine_bpi": fine_bpi.astype(float),
        "broad_std": broad_std.astype(float),
        "fine_std": fine_std.astype(float),
        "slope": slope.astype(float),
        "vrm": vrm.astype(float),
        "surface_ratio": surf_ratio.astype(float),
    }

    if include_rule_class and classification_file is not None:
        from btm.classification.reader import read_classification

        classes = read_classification(str(classification_file))
        rule_class = classify_terrain(broad_std, fine_std, slope, bathy, classes)
        derivatives["rule_class"] = rule_class.astype(float)

    # ------------------------------------------------------------------
    # 3. Optionally write derivative rasters to disk
    # ------------------------------------------------------------------
    if outdir is not None:
        _write_derivatives(bathy_ds, derivatives, Path(outdir))

    # ------------------------------------------------------------------
    # 4. Sample derivatives at (x, y) point locations
    # ------------------------------------------------------------------
    xs = sample_points["x"].to_numpy(dtype=float)
    ys = sample_points["y"].to_numpy(dtype=float)

    result = sample_points.copy()
    for name, array in derivatives.items():
        col = f"{btm_prefix}{name}"
        result[col] = sample_raster_at_points(array, transform, xs, ys, nodata=nodata)

    # ------------------------------------------------------------------
    # 5. Interaction features
    # ------------------------------------------------------------------
    if include_interactions:
        p = btm_prefix
        bs = result[f"{p}broad_std"]
        fs = result[f"{p}fine_std"]
        vr = result[f"{p}vrm"]
        sr = result[f"{p}surface_ratio"]

        # BPI magnitude: how strongly is this location elevated/depressed
        # at both scales?  Useful for crests (high broad, high fine) and
        # nested depressions (low broad, low fine) which rule-based thresholds
        # treat as separate decision regions.
        result[f"{p}bpi_magnitude"] = np.sqrt(bs**2 + fs**2)

        # Cross-scale BPI product: positive = same sign at both scales
        # (clear crest or clear depression); negative = dissimilar scales
        # (ridge sitting in a larger depression, etc.)
        result[f"{p}broad_x_fine_std"] = bs * fs

        # Combined roughness (VRM is [0,1], surface_ratio is ≥ 1.0 — centre
        # surface_ratio around 1 so both terms are zero for a flat surface)
        result[f"{p}rough_total"] = vr + (sr - 1.0)

    _log.info(
        "extract_btm_features: produced %d BTM columns for %d points",
        sum(1 for c in result.columns if c.startswith(btm_prefix)),
        len(result),
    )

    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_derivatives(
    template_ds,
    derivatives: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for name, array in derivatives.items():
        path = outdir / f"btm_{name}.tif"
        # Use float32 for all intermediate writes
        template_ds.array = array.astype(np.float32)
        template_ds.to_file(str(path), dtype="float32")
        _log.debug("wrote %s", path)
