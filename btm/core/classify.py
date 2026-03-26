"""Terrain classification via sequential CON cascade.

Preserves the BTM v3.0 first-match-wins conditional expression exactly.
"""

from __future__ import annotations

import numpy as np

from btm.classification.reader import ClassEntry


def classify_terrain(
    broad_std: np.ndarray,
    fine_std: np.ndarray,
    slope: np.ndarray,
    bathy: np.ndarray,
    classes: list[ClassEntry],
    nodata: float | None = None,
) -> np.ndarray:
    """Classify benthic terrain.

    Iterates over *classes* in order (CON cascade); the first matching class
    for each cell wins.  Unclassified cells remain 0.

    Parameters
    ----------
    broad_std:
        Standardised broad-scale BPI array.
    fine_std:
        Standardised fine-scale BPI array.
    slope:
        Slope array in degrees.
    bathy:
        Original bathymetric array.
    classes:
        Ordered list of ClassEntry objects.
    nodata:
        If provided, cells where *any* input equals this value are marked
        as nodata (set to 0 in output; callers can treat 0 as nodata).

    Returns
    -------
    np.ndarray
        int32 array of class codes; 0 = unclassified / nodata.
    """
    shape = broad_std.shape
    result = np.zeros(shape, dtype=np.int32)

    # Build a nodata mask (True where output should be forced to 0)
    if nodata is not None:
        nodata_cells = (
            (broad_std == nodata)
            | (fine_std == nodata)
            | (slope == nodata)
            | (bathy == nodata)
        )
    else:
        nodata_cells = np.zeros(shape, dtype=bool)

    for cls in classes:
        mask = np.ones(shape, dtype=bool)

        if cls.broad_bpi_lower is not None:
            mask &= broad_std >= cls.broad_bpi_lower
        if cls.broad_bpi_upper is not None:
            mask &= broad_std < cls.broad_bpi_upper
        if cls.fine_bpi_lower is not None:
            mask &= fine_std >= cls.fine_bpi_lower
        if cls.fine_bpi_upper is not None:
            mask &= fine_std < cls.fine_bpi_upper
        if cls.slope_lower is not None:
            mask &= slope >= cls.slope_lower
        if cls.slope_upper is not None:
            mask &= slope < cls.slope_upper
        if cls.depth_lower is not None:
            mask &= bathy >= cls.depth_lower
        if cls.depth_upper is not None:
            mask &= bathy < cls.depth_upper

        # First-match-wins: only assign where result is still 0
        result = np.where(mask & (result == 0), cls.code, result)

    # Zero out nodata cells
    result[nodata_cells] = 0

    return result.astype(np.int32)
