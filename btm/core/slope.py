"""Slope computation using the Horn (1981) 3×3 weighted finite-difference kernel.

Reference: Horn, B.K.P. 1981. Hill shading and the reflectance map.
           Proceedings of the IEEE, 69(1): 14–47.
Formula:   slope_deg = arctan(sqrt((dz/dx)² + (dz/dy)²)) × 180/π

3×3 neighbourhood labelling (row-major):
    a  b  c
    d  e  f
    g  h  i

Horn kernel:
    dz/dx = ((c + 2f + i) − (a + 2d + g)) / (8 × cell_size)
    dz/dy = ((a + 2b + c) − (g + 2h + i)) / (8 × cell_size)
"""

from __future__ import annotations

import numpy as np


def compute_slope(
    array: np.ndarray,
    cell_size: float,
    nodata: float | None = None,
) -> np.ndarray:
    """Compute slope in degrees (Horn 1981).

    Parameters
    ----------
    array:
        2-D elevation / bathymetry array.
    cell_size:
        Cell size in CRS units.
    nodata:
        If provided, nodata cells are masked during gradient computation
        and propagated to the output.

    Returns
    -------
    np.ndarray
        float32 array of slope values in degrees [0, 90].
    """
    work = array.astype(np.float64)

    if nodata is not None:
        nodata_mask = work == nodata
        work[nodata_mask] = np.nan
    else:
        nodata_mask = None

    # Pad one cell on each side (reflect mode) to handle edge cells
    padded = np.pad(work, 1, mode="reflect")

    # Extract 3×3 neighbourhood arrays (all shifted views of padded)
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    # e = padded[1:-1, 1:-1]  # centre — not needed
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]

    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell_size)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * cell_size)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)

    if nodata_mask is not None:
        slope_deg[nodata_mask] = np.nan

    return slope_deg.astype(np.float32)


def _compute_gradient(array: np.ndarray, cell_size: float):
    """Return (dz_dx, dz_dy) using the Horn kernel.  Internal helper for VRM."""
    padded = np.pad(array.astype(np.float64), 1, mode="reflect")
    a = padded[:-2, :-2]
    b = padded[:-2, 1:-1]
    c = padded[:-2, 2:]
    d = padded[1:-1, :-2]
    f = padded[1:-1, 2:]
    g = padded[2:, :-2]
    h = padded[2:, 1:-1]
    i = padded[2:, 2:]
    dz_dx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * cell_size)
    dz_dy = ((a + 2 * b + c) - (g + 2 * h + i)) / (8.0 * cell_size)
    return dz_dx, dz_dy
