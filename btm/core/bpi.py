"""Bathymetric Position Index (BPI) computation.

Reference: Wright, D.J. et al. 2005. ArcGIS Benthic Terrain Modeler.
Formula:   BPI = round(bathy − focal_annular_mean(inner_radius, outer_radius))
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def _build_annulus(inner_radius: int, outer_radius: int) -> np.ndarray:
    """Return a boolean 2-D footprint mask for an annular neighbourhood."""
    y, x = np.ogrid[-outer_radius : outer_radius + 1, -outer_radius : outer_radius + 1]
    dist = np.sqrt(x**2 + y**2)
    mask = (dist >= inner_radius) & (dist <= outer_radius)
    return mask.astype(bool)


def compute_bpi(
    array: np.ndarray,
    inner_radius: int,
    outer_radius: int,
    cell_size: float,
    nodata: float | None = None,
) -> np.ndarray:
    """Compute Bathymetric Position Index.

    Parameters
    ----------
    array:
        2-D input bathymetry array.
    inner_radius:
        Inner radius of the annulus in cells (exclusive).
    outer_radius:
        Outer radius of the annulus in cells (inclusive).
    cell_size:
        Cell size in CRS units (unused in computation, kept for API symmetry).
    nodata:
        If provided, cells equal to this value are excluded from the focal mean
        and propagated to the output.

    Returns
    -------
    np.ndarray
        int32 array of BPI values.

    Raises
    ------
    ValueError
        If inner_radius >= outer_radius or outer_radius < 1.
    """
    if outer_radius < 1:
        raise ValueError("outer_radius must be >= 1.")
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be strictly less than outer_radius.")

    footprint = _build_annulus(inner_radius, outer_radius)

    work = array.astype(np.float64)

    if nodata is not None:
        nodata_mask = work == nodata
        work[nodata_mask] = np.nan
    else:
        nodata_mask = None

    def _annular_mean(values: np.ndarray) -> float:
        valid = values[~np.isnan(values)]
        return float(np.mean(valid)) if valid.size > 0 else np.nan

    focal_mean = scipy.ndimage.generic_filter(
        work, _annular_mean, footprint=footprint, mode="reflect"
    )

    diff = work - focal_mean
    result = np.round(diff).astype(np.float64)

    if nodata is not None:
        result[nodata_mask] = nodata  # type: ignore[index]
        return result.astype(np.int32)

    return result.astype(np.int32)
