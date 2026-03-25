"""BPI standardisation.

Reference: BTM v3.0 standardize_bpi_grids.py
Formula:   stdBPI = round((bpi − mean) / std × 100)
"""

from __future__ import annotations

import numpy as np


def standardize_bpi(
    array: np.ndarray,
    nodata: float | None = None,
) -> np.ndarray:
    """Standardise a BPI raster.

    Parameters
    ----------
    array:
        2-D BPI array.
    nodata:
        If provided, cells equal to this value are excluded from mean/std
        and propagated to the output.

    Returns
    -------
    np.ndarray
        int32 array of standardised values.

    Raises
    ------
    ValueError
        If the standard deviation of valid cells is 0.
    """
    work = array.astype(np.float64)

    if nodata is not None:
        valid_mask = work != nodata
    else:
        valid_mask = np.ones(work.shape, dtype=bool)

    valid = work[valid_mask]
    if valid.size == 0:
        return np.zeros_like(array, dtype=np.int32)

    mean = float(np.mean(valid))
    std = float(np.std(valid))

    if std == 0.0:
        raise ValueError(
            "Standard deviation of valid BPI cells is 0 — cannot standardise a "
            "constant raster."
        )

    result = np.round((work - mean) / std * 100).astype(np.int32)

    if nodata is not None:
        result[~valid_mask] = int(nodata)

    return result
