"""Surface-area-to-planar-area ratio (Jenness rugosity).

Reference: Jenness, J. 2002. Surface Areas and Ratios from Elevation Grid
           (surfgrids.avx) extension for ArcView 3.x, v1.2.
           Jenness Enterprises.

Vectorised NumPy port of the shifted-array approach from the BTM v3.0
`surface_area_to_planar_area.py` script (arcpy removed).

Neighbourhood layout (1-indexed positions matching original):
        8 | 7 | 6
        --|---|---
        5 | X | 4
        --|---|---
        3 | 2 | 1

Edges connect X to each of 8 neighbours. 8 triangles are formed:
    (X,1,2), (X,2,3), (X,3,5), (X,5,8), (X,8,7), (X,7,6), (X,6,4), (X,4,1)
"""

from __future__ import annotations

import math

import numpy as np


def _edge(r1: np.ndarray, r2: np.ndarray, dist: float) -> np.ndarray:
    r"""Edge length: sqrt((r1 - r2)^2 + dist^2) / 2."""
    return np.sqrt((r1 - r2) ** 2 + dist**2) / 2.0


def _triangle_area(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Heron-style triangle area (Jenness pp 11–12)."""
    inner = b**2 - ((a**2 + b**2 - c**2) / (2.0 * a)) ** 2
    inner = np.maximum(inner, 0.0)  # guard numerical noise
    return a * np.sqrt(inner) / 2.0


def compute_surface_planar_ratio(
    array: np.ndarray,
    cell_size: float,
    nodata: float | None = None,
) -> np.ndarray:
    """Compute surface-area-to-planar-area ratio.

    Parameters
    ----------
    array:
        2-D elevation / bathymetry array.
    cell_size:
        Mean cell size in CRS units.
    nodata:
        NoData sentinel value.

    Returns
    -------
    np.ndarray
        float32 array of ratio values (≥ 1.0 for non-flat surfaces;
        border cells may differ).
    """
    work = array.astype(np.float64)

    if nodata is not None:
        nodata_mask: np.ndarray = work == nodata
        work[nodata_mask] = np.nan
    else:
        nodata_mask = None  # type: ignore[assignment]

    corner_dist = math.sqrt(2.0) * cell_size
    flat_area = cell_size**2

    # Shifted grids: positions 1–8 around X (centre)
    # Padding with edge replication
    padded = np.pad(work, 1, mode="edge")

    # Position layout (relative to centre padded[1:-1,1:-1] = X):
    #  8(r-1,c-1)  7(r-1,c)  6(r-1,c+1)
    #  5(r,  c-1)  X(r,  c)  4(r,  c+1)
    #  3(r+1,c-1)  2(r+1,c)  1(r+1,c+1)
    X = padded[1:-1, 1:-1]
    s1 = padded[2:, 2:]  # bottom-right
    s2 = padded[2:, 1:-1]  # bottom
    s3 = padded[2:, :-2]  # bottom-left
    s4 = padded[1:-1, 2:]  # right
    s5 = padded[1:-1, :-2]  # left
    s6 = padded[:-2, 2:]  # top-right
    s7 = padded[:-2, 1:-1]  # top
    s8 = padded[:-2, :-2]  # top-left

    # Edges from X to each neighbour
    e_x1 = _edge(X, s1, corner_dist)
    e_x2 = _edge(X, s2, cell_size)
    e_x3 = _edge(X, s3, corner_dist)
    e_x4 = _edge(X, s4, cell_size)
    e_x5 = _edge(X, s5, cell_size)
    e_x6 = _edge(X, s6, corner_dist)
    e_x7 = _edge(X, s7, cell_size)
    e_x8 = _edge(X, s8, corner_dist)

    # Edges between adjacent neighbours (sides of triangles).
    # All adjacent pairs in the Jenness layout are orthogonally adjacent
    # (one cell apart in either row or column) → 2-D distance = cell_size.
    e_12 = _edge(s1, s2, cell_size)
    e_23 = _edge(s2, s3, cell_size)
    e_35 = _edge(s3, s5, cell_size)
    e_58 = _edge(s5, s8, cell_size)
    e_87 = _edge(s8, s7, cell_size)
    e_76 = _edge(s7, s6, cell_size)
    e_64 = _edge(s6, s4, cell_size)
    e_41 = _edge(s4, s1, cell_size)

    # Sum of 8 triangle areas
    surface_area = (
        _triangle_area(e_x1, e_x2, e_12)
        + _triangle_area(e_x2, e_x3, e_23)
        + _triangle_area(e_x3, e_x5, e_35)
        + _triangle_area(e_x5, e_x8, e_58)
        + _triangle_area(e_x8, e_x7, e_87)
        + _triangle_area(e_x7, e_x6, e_76)
        + _triangle_area(e_x6, e_x4, e_64)
        + _triangle_area(e_x4, e_x1, e_41)
    )

    ratio = surface_area / flat_area

    if nodata_mask is not None:
        ratio[nodata_mask] = np.nan

    return ratio.astype(np.float32)
