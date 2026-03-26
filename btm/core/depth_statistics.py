"""Focal depth statistics (mean, std, variance, IQR, kurtosis).

Ports the block-processing approach from the BTM v3.0 depth_statistics.py,
replacing arcpy / netCDF4 with rasterio and pure NumPy/SciPy.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import scipy.ndimage
import scipy.stats

_VALID_STATS = frozenset(["mean", "std", "variance", "iqr", "kurtosis", "mean_diff"])


def _focal_iqr(array: np.ndarray, n_size: int) -> np.ndarray:
    """Compute focal IQR using a stacked-neighbourhood approach."""
    overlap = n_size // 2
    s = array.shape
    rows = 2 * overlap + 1
    cols = 2 * overlap + 1
    nbh_list = []
    for r in range(rows):
        for c in range(cols):
            nbh_list.append(array[r : s[0] - (rows - 1) + r, c : s[1] - (cols - 1) + c])
    stack = np.array(nbh_list)
    result_inner = np.percentile(stack, 75, axis=0) - np.percentile(stack, 25, axis=0)
    # Pad back to original shape (edges will be NaN)
    pad_h = overlap
    pad_w = overlap
    out = np.full(s, np.nan, dtype=np.float64)
    out[pad_h : s[0] - pad_h, pad_w : s[1] - pad_w] = result_inner
    return out


def _focal_kurtosis(array: np.ndarray, n_size: int) -> np.ndarray:
    """Compute focal kurtosis using a stacked-neighbourhood approach."""
    overlap = n_size // 2
    s = array.shape
    rows = 2 * overlap + 1
    cols = 2 * overlap + 1
    nbh_list = []
    for r in range(rows):
        for c in range(cols):
            nbh_list.append(array[r : s[0] - (rows - 1) + r, c : s[1] - (cols - 1) + c])
    stack = np.array(nbh_list)
    result_inner = scipy.stats.kurtosis(stack, axis=0)
    pad_h = overlap
    pad_w = overlap
    out = np.full(s, np.nan, dtype=np.float64)
    out[pad_h : s[0] - pad_h, pad_w : s[1] - pad_w] = result_inner
    return out


def compute_focal_stats(
    array: np.ndarray,
    n_size: int,
    stats: Sequence[str],
    window_type: str = "rectangle",
    nodata: float | None = None,
) -> dict[str, np.ndarray]:
    """Compute one or more focal depth statistics.

    Parameters
    ----------
    array:
        2-D bathymetric array.
    n_size:
        Neighbourhood size in cells (must be ≥ 3).
    stats:
        List of requested stats.  Supported values: ``mean``, ``std``,
        ``variance``, ``iqr``, ``kurtosis``, ``mean_diff``.
    window_type:
        ``'rectangle'`` (default) or ``'circle'``.  Circle uses a
        disc-shaped footprint; rectangle is a square n×n window.
    nodata:
        NoData sentinel to exclude from computations.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of stat name → float64 result array.

    Raises
    ------
    ValueError
        If an unsupported stat name is provided.
    """
    unknown = set(stats) - _VALID_STATS
    if unknown:
        raise ValueError(f"Unknown stat name(s): {', '.join(sorted(unknown))}")

    work = array.astype(np.float64)
    if nodata is not None:
        work = np.where(work == nodata, np.nan, work)

    # Build footprint for circle window
    if window_type == "circle":
        r = n_size // 2
        y, x = np.ogrid[-r : r + 1, -r : r + 1]
        footprint: np.ndarray | None = (x**2 + y**2 <= r**2).astype(bool)
    else:
        footprint = None  # use size= parameter for rectangle

    results: dict[str, np.ndarray] = {}

    def _uniform(arr: np.ndarray) -> np.ndarray:
        if footprint is not None:
            return scipy.ndimage.generic_filter(
                arr, lambda v: np.nanmean(v), footprint=footprint, mode="reflect"
            )
        return scipy.ndimage.uniform_filter(arr, size=n_size, mode="reflect")

    if "mean" in stats or "mean_diff" in stats or "variance" in stats:
        focal_mean = _uniform(work)
        if "mean" in stats:
            results["mean"] = focal_mean

    if "std" in stats or "variance" in stats:
        # Var(X) = E[X²] - E[X]²
        focal_mean2 = _uniform(work**2)
        focal_var = focal_mean2 - focal_mean**2
        focal_var = np.maximum(focal_var, 0.0)  # guard floating-point negatives
        focal_std = np.sqrt(focal_var)
        if "std" in stats:
            results["std"] = focal_std
        if "variance" in stats:
            results["variance"] = focal_var

    if "mean_diff" in stats:
        focal_range = _uniform(np.abs(work - focal_mean))
        with np.errstate(divide="ignore", invalid="ignore"):
            mean_diff = np.where(
                focal_range != 0, -(focal_mean - work) / focal_range, 0.0
            )
        results["mean_diff"] = mean_diff

    if "iqr" in stats:
        results["iqr"] = _focal_iqr(work, n_size)

    if "kurtosis" in stats:
        results["kurtosis"] = _focal_kurtosis(work, n_size)

    return results
