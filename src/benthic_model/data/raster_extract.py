from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def sample_raster_values(
    points: pd.DataFrame,
    raster_path: str | Path,
    prefix: str,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Sample a single-band raster at point locations.

    Returns a copy of *points* with two new columns added:
        ``<prefix>``       – sampled float value (NaN for nodata / out-of-bounds)
        ``<prefix>_flag``  – "" | "nodata" | "out_of_bounds"
    """
    result = points.copy()
    values = np.full(len(points), np.nan, dtype=float)
    flags = np.full(len(points), "", dtype=object)

    with rasterio.open(raster_path) as src:
        nodata = src.nodata
        xs = points[x_col].to_numpy(dtype=float)
        ys = points[y_col].to_numpy(dtype=float)

        rows, cols = rasterio.transform.rowcol(src.transform, xs, ys)
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        height, width = src.height, src.width
        in_bounds = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)

        flags[~in_bounds] = "out_of_bounds"

        if in_bounds.any():
            data = src.read(1)
            sampled = data[rows[in_bounds], cols[in_bounds]].astype(float)
            if nodata is not None:
                is_nodata = sampled == float(nodata)
                sampled[is_nodata] = np.nan
                flag_indices = np.where(in_bounds)[0][is_nodata]
                flags[flag_indices] = "nodata"
            values[in_bounds] = sampled

    result[prefix] = values
    result[f"{prefix}_flag"] = flags
    return result


def extract_mbes_features(
    points: pd.DataFrame,
    bathymetry_tif: str | Path,
    backscatter_tif: str | Path,
) -> pd.DataFrame:
    """Extract bathymetry and backscatter raster values at sample point locations.

    Args:
        points: DataFrame with at least ``ID``, ``x``, ``y`` columns.
        bathymetry_tif: Path to the bathymetry GeoTIFF.
        backscatter_tif: Path to the backscatter GeoTIFF.

    Returns:
        DataFrame with ``bathymetry``, ``bathymetry_flag``,
        ``backscatter``, and ``backscatter_flag`` columns appended.
    """
    result = sample_raster_values(points, bathymetry_tif, prefix="bathymetry")
    result = sample_raster_values(result, backscatter_tif, prefix="backscatter")
    return result
