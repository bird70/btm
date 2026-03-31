"""Vector Ruggedness Measure (VRM).

Reference: Sappington, J.M., K.M. Longshore, D.B. Thompson. 2007.
           Quantifying Landscape Ruggedness for Animal Habitat Analysis:
           A Case Study Using Bighorn Sheep in the Mojave Desert.
           Journal of Wildlife Management, 71(5): 1419–1426.

Formula:
    1. slope_rad = slope_deg × π/180
    2. aspect_rad = aspect_deg × π/180   (ArcGIS convention: 0=N, clockwise)
    3. xy = sin(slope_rad)
    4. x  = where(aspect == -1, 0, sin(aspect_rad)) × xy
    5. y  = where(aspect == -1, 0, cos(aspect_rad)) × xy
    6. z  = cos(slope_rad)
    7. Focal sum of x, y, z over n×n square neighbourhood
    8. resultant = sqrt(x_sum² + y_sum² + z_sum²)
    9. VRM = 1 − resultant / n²
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from btm.core.slope import _compute_gradient


def compute_vrm(
    array: np.ndarray,
    neighborhood_size: int,
    cell_size: float,
    nodata: float | None = None,
) -> np.ndarray:
    """Compute the Vector Ruggedness Measure (Sappington et al. 2007).

    Parameters
    ----------
    array:
        2-D elevation / bathymetry array.
    neighborhood_size:
        Square neighbourhood size in cells (must be ≥ 3 and odd).
    cell_size:
        Cell size in CRS units.
    nodata:
        NoData sentinel value.

    Returns
    -------
    np.ndarray
        float32 array of VRM values in [0, 1].

    Raises
    ------
    ValueError
        If neighbourhood_size < 3.
    """
    if neighborhood_size < 3:
        raise ValueError(f"neighborhood_size must be >= 3, got {neighborhood_size}.")

    work = array.astype(np.float64)

    if nodata is not None:
        nodata_mask: np.ndarray = work == nodata
        work[nodata_mask] = np.nan
    else:
        nodata_mask = None  # type: ignore[assignment]

    # scipy.ndimage.uniform_filter propagates NaN to all neighbours.
    # Fill nodata pixels with 0 for the focal-sum step, then restore NaN mask.
    nan_mask = np.isnan(work)
    work_filled = np.where(nan_mask, 0.0, work)

    # --- Slope and aspect via Horn kernel ---
    dz_dx, dz_dy = _compute_gradient(work_filled, cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    # ArcGIS-style aspect: 0=N, clockwise; flat cells → -1
    aspect_rad = np.arctan2(-dz_dy, dz_dx)  # math convention first
    # Convert math→compass: 90° - atan2 then normalise [0, 360)
    aspect_compass = np.degrees(aspect_rad)
    aspect_compass = 90.0 - aspect_compass
    aspect_compass = np.where(
        aspect_compass < 0, aspect_compass + 360.0, aspect_compass
    )
    # Flat cells (dz/dx == 0 and dz/dy == 0) → aspect = -1
    flat = (dz_dx == 0) & (dz_dy == 0)
    aspect_deg = np.where(flat, -1.0, aspect_compass)
    aspect_r = np.radians(np.where(flat, 0.0, aspect_deg))

    # --- Unit vector components ---
    xy = np.sin(slope_rad)
    x = np.where(aspect_deg == -1, 0.0, np.sin(aspect_r)) * xy
    y = np.where(aspect_deg == -1, 0.0, np.cos(aspect_r)) * xy
    z = np.cos(slope_rad)

    # Replace NaN with 0 before focal sum so uniform_filter doesn't propagate NaN
    x = np.where(nan_mask, 0.0, x)
    y = np.where(nan_mask, 0.0, y)
    z = np.where(nan_mask, 0.0, z)

    # --- Focal sum via uniform_filter (gives mean → multiply by n²) ---
    n2 = float(neighborhood_size**2)
    x_sum = scipy.ndimage.uniform_filter(x, size=neighborhood_size, mode="reflect") * n2
    y_sum = scipy.ndimage.uniform_filter(y, size=neighborhood_size, mode="reflect") * n2
    z_sum = scipy.ndimage.uniform_filter(z, size=neighborhood_size, mode="reflect") * n2

    resultant = np.sqrt(x_sum**2 + y_sum**2 + z_sum**2)
    vrm = 1.0 - resultant / n2

    # Clamp to [0, 1] to remove floating-point overshoot
    vrm = np.clip(vrm, 0.0, 1.0)

    if nodata_mask is not None:
        vrm[nodata_mask] = np.nan

    return vrm.astype(np.float32)
