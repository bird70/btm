"""GLCM texture feature extraction from backscatter rasters.

Grey-Level Co-occurrence Matrix (GLCM) features are computed per-point by
extracting a square patch from the backscatter raster centred on each point,
averaging the GLCM properties over four directions (0°, 45°, 90°, 135°) for
rotation invariance.

References
----------
Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973).
    Textural features for image classification.
    *IEEE Transactions on Systems, Man, and Cybernetics*, 3(6), 610–621.

Nemani, V. P. et al. (2022).
    A multi-scale feature selection approach for predicting benthic
    assemblages.  *ICES Journal of Marine Science*, 79(2), 547–562.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_glcm_texture(
    backscatter_array: np.ndarray,
    point_rows: np.ndarray,
    point_cols: np.ndarray,
    scales: list[int],
    n_levels: int = 32,
) -> pd.DataFrame:
    """Extract GLCM contrast and homogeneity at multiple window sizes.

    For each point location, a square patch of size ``scale × scale`` is
    extracted from ``backscatter_array``, centred on ``(row, col)``.  Any NaN
    pixels in the patch are replaced by the patch median before computing the
    GLCM.  The GLCM is computed with ``distances=[1]`` and
    ``angles=[0, π/4, π/2, 3π/4]``, then contrast and homogeneity are
    averaged over the four directions for rotation invariance.

    This implementation promotes and standardises the pattern found in
    ``scripts/experiment_v2.py`` and ``scripts/experiment_v5.py``.

    Time complexity: O(p · w² · L²) where p = number of points, w = window
    size, L = ``n_levels`` (grey-level quantisation depth).
    Space complexity: O(L²) per point.

    Parameters
    ----------
    backscatter_array:
        2-D NumPy array of backscatter values (float32 or float64).
        NaN / nodata values are handled by median imputation within the patch.
    point_rows:
        1-D integer array of row indices (0-based) for each point.
    point_cols:
        1-D integer array of column indices (0-based) for each point.
    scales:
        List of odd window sizes (in pixels) at which to compute GLCM,
        e.g. ``[7, 11, 15]``.
    n_levels:
        Number of grey-level bins for GLCM quantisation (default: 32).

    Returns
    -------
    pd.DataFrame
        Shape ``(len(point_rows), 2 × len(scales))``.  Columns follow the
        naming convention ``btm_glcm_contrast_{scale}`` and
        ``btm_glcm_homogeneity_{scale}``.

    References
    ----------
    Haralick et al. (1973) and Nemani et al. (2022) — see module docstring.
    """
    from skimage.feature import graycomatrix, graycoprops

    n_pts = len(point_rows)
    rows_arr = np.asarray(point_rows, dtype=int)
    cols_arr = np.asarray(point_cols, dtype=int)
    nrows, ncols = backscatter_array.shape

    results: dict[str, np.ndarray] = {}
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    for scale in scales:
        half = scale // 2
        contrast_vals = np.full(n_pts, np.nan, dtype=np.float64)
        homogeneity_vals = np.full(n_pts, np.nan, dtype=np.float64)

        for i in range(n_pts):
            r, c = rows_arr[i], cols_arr[i]

            # Compute patch boundaries, clamped to raster extent
            r0 = max(0, r - half)
            r1 = min(nrows, r + half + 1)
            c0 = max(0, c - half)
            c1 = min(ncols, c + half + 1)

            if r1 <= r0 or c1 <= c0:
                continue

            patch = backscatter_array[r0:r1, c0:c1].astype(np.float64)

            # Replace NaN with patch median
            finite_mask = np.isfinite(patch)
            if not finite_mask.any():
                continue
            if not finite_mask.all():
                patch_median = float(np.nanmedian(patch))
                patch[~finite_mask] = patch_median

            # Quantise to [0, n_levels - 1]
            p_min, p_max = patch.min(), patch.max()
            if p_max == p_min:
                # Uniform patch — no texture
                contrast_vals[i] = 0.0
                homogeneity_vals[i] = 1.0
                continue

            quantised = ((patch - p_min) / (p_max - p_min) * (n_levels - 1)).astype(
                np.uint8
            )

            glcm = graycomatrix(
                quantised,
                distances=[1],
                angles=angles,
                levels=n_levels,
                symmetric=True,
                normed=True,
            )

            # Average over 4 directions
            contrast_vals[i] = float(graycoprops(glcm, "contrast").mean())
            homogeneity_vals[i] = float(graycoprops(glcm, "homogeneity").mean())

        results[f"btm_glcm_contrast_{scale}"] = contrast_vals
        results[f"btm_glcm_homogeneity_{scale}"] = homogeneity_vals

    return pd.DataFrame(results)
