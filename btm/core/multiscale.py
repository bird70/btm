"""Multi-scale terrain feature computation via calculate-then-average.

Each terrain derivative (slope, VRM, surface ratio, northness, eastness,
max curvature, complexity) is first computed at the native 3×3 kernel
resolution, then spatially averaged at multiple neighbourhood window sizes
using ``scipy.ndimage.uniform_filter``.  The Relative Difference to Mean
Value (RDMV) is additionally computed directly from the bathymetry grid at
each scale.

References
----------
Misiuk, B., Lecours, V., Dolan, M. F. J., & Robert, K. (2021).
    Evaluating the suitability of multi-scale terrain attribute calculation
    approaches for seabed mapping applications.
    *ISPRS International Journal of Geo-Information*, 10(4), 193.

Lecours, V., Brown, C. J., Devillers, R., Lucieer, V. L., & Edinger, E. N.
    (2017). Comparing selections of environmental variables for ecological
    studies: a focus on terrain attributes.
    *PLOS ONE*, 12(10), e0186165.
"""

from __future__ import annotations

import numpy as np
import scipy.ndimage


def multiscale_column_names(derivative: str, scales: list[int]) -> list[str]:
    """Return the standardised column names for a multi-scale derivative.

    Parameters
    ----------
    derivative:
        Derivative identifier, e.g. ``"slope"`` or ``"vrm"``.
    scales:
        List of neighbourhood window sizes, e.g. ``[3, 7, 11, 15, 21]``.

    Returns
    -------
    list[str]
        Column names following the convention ``btm_{derivative}_{scale}``.

    Examples
    --------
    >>> multiscale_column_names("slope", [3, 7])
    ['btm_slope_3', 'btm_slope_7']
    """
    return [f"btm_{derivative}_{s}" for s in scales]


def focal_mean_multiscale(
    derivative_name: str,
    base_array: np.ndarray,
    scales: list[int],
) -> dict[int, np.ndarray]:
    """Compute the spatial focal mean of a derivative at multiple window sizes.

    Applies ``scipy.ndimage.uniform_filter`` with ``mode='reflect'`` at each
    requested scale, returning one averaged array per scale.  This implements
    the *calculate-then-average* multi-scale strategy described in Misiuk et
    al. (2021): compute the derivative at native 3×3 resolution first, then
    smooth over progressively larger neighbourhoods to capture broader-scale
    terrain context.

    Time complexity: O(N) per scale (separable box filter, where N is the
    total number of pixels).  Space complexity: O(N) additional per scale.

    Parameters
    ----------
    derivative_name:
        Identifier for logging (e.g. ``"slope"``); not used in computation.
    base_array:
        2-D NumPy array containing the native-resolution derivative values.
        May be float32 or float64.
    scales:
        List of odd integer window sizes (e.g. ``[3, 7, 11, 15, 21]``).
        Each value is passed directly as ``size`` to ``uniform_filter``.

    Returns
    -------
    dict[int, np.ndarray]
        Mapping from scale → focal-averaged array (float64), same spatial
        shape as ``base_array``.

    References
    ----------
    Misiuk et al. (2021) — see module docstring.
    """
    work = base_array.astype(np.float64)
    result: dict[int, np.ndarray] = {}
    for s in scales:
        result[s] = scipy.ndimage.uniform_filter(work, size=s, mode="reflect")
    return result


def compute_rdmv(
    depth_array: np.ndarray,
    scale: int,
) -> np.ndarray:
    """Compute Relative Difference to Mean Value (RDMV) at a given scale.

    RDMV measures how different a cell's depth value is from its local
    neighbourhood mean, normalised by the neighbourhood standard deviation::

        RDMV = (depth − focal_mean) / focal_std

    Positive values indicate a cell is above (shallower than) its local mean
    (e.g. a ridge); negative values indicate a depression.  Flat areas where
    ``focal_std == 0`` are assigned 0.0 rather than NaN.

    Time complexity: O(N) — two separable box-filter passes.
    Space complexity: O(N) additional.

    Parameters
    ----------
    depth_array:
        2-D NumPy array of depth / elevation values.
    scale:
        Neighbourhood window size in cells (odd integer ≥ 3).

    Returns
    -------
    np.ndarray
        float64 array of RDMV values, same shape as ``depth_array``.
        Zero where ``focal_std == 0``.

    References
    ----------
    Lecours, V. et al. (2017) — see module docstring.
    """
    work = depth_array.astype(np.float64)

    focal_mean = scipy.ndimage.uniform_filter(work, size=scale, mode="reflect")

    # Variance via E[X²] − (E[X])²; clip to zero to suppress floating-point noise
    focal_mean_sq = scipy.ndimage.uniform_filter(work**2, size=scale, mode="reflect")
    variance = np.maximum(focal_mean_sq - focal_mean**2, 0.0)
    focal_std = np.sqrt(variance)

    with np.errstate(invalid="ignore", divide="ignore"):
        rdmv = np.where(focal_std == 0.0, 0.0, (work - focal_mean) / focal_std)
    return rdmv
