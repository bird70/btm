"""Unit tests for btm.core.glcm — GLCM texture feature extraction."""

from __future__ import annotations

import numpy as np
import pytest

from btm.core.glcm import compute_glcm_texture


def _make_points(n: int, arr_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return (rows, cols) for n evenly spaced interior points."""
    nrows, ncols = arr_shape
    rows = np.linspace(nrows // 4, 3 * nrows // 4, n, dtype=int)
    cols = np.linspace(ncols // 4, 3 * ncols // 4, n, dtype=int)
    return rows, cols


class TestComputeGlcmTexture:
    def test_output_columns(self):
        """Output DataFrame must have the correct column names and row count."""
        rng = np.random.default_rng(0)
        arr = rng.random((40, 40)).astype(np.float32)
        rows, cols = _make_points(5, arr.shape)
        df = compute_glcm_texture(arr, rows, cols, scales=[7])
        assert list(df.columns) == ["btm_glcm_contrast_7", "btm_glcm_homogeneity_7"]
        assert len(df) == 5

    def test_multi_scale_columns(self):
        rng = np.random.default_rng(1)
        arr = rng.random((40, 40)).astype(np.float32)
        rows, cols = _make_points(3, arr.shape)
        df = compute_glcm_texture(arr, rows, cols, scales=[7, 11])
        expected_cols = [
            "btm_glcm_contrast_7",
            "btm_glcm_homogeneity_7",
            "btm_glcm_contrast_11",
            "btm_glcm_homogeneity_11",
        ]
        assert list(df.columns) == expected_cols

    def test_contrast_nonnegative(self):
        """GLCM contrast is always ≥ 0."""
        rng = np.random.default_rng(2)
        arr = rng.random((60, 60)).astype(np.float32)
        rows, cols = _make_points(10, arr.shape)
        df = compute_glcm_texture(arr, rows, cols, scales=[9])
        assert (df["btm_glcm_contrast_9"].dropna() >= 0.0).all()

    def test_homogeneity_range(self):
        """GLCM homogeneity must lie in [0, 1]."""
        rng = np.random.default_rng(3)
        arr = rng.random((60, 60)).astype(np.float32)
        rows, cols = _make_points(10, arr.shape)
        df = compute_glcm_texture(arr, rows, cols, scales=[9])
        h = df["btm_glcm_homogeneity_9"].dropna()
        assert (h >= 0.0).all()
        assert (h <= 1.0).all()

    def test_nodata_in_patch_returns_finite(self):
        """NaN pixels in the patch are imputed; output must be finite."""
        rng = np.random.default_rng(4)
        arr = rng.random((40, 40)).astype(np.float64)
        # Inject NaN values in the patch region
        arr[18:22, 18:22] = np.nan
        rows = np.array([20])
        cols = np.array([20])
        df = compute_glcm_texture(arr, rows, cols, scales=[7])
        assert np.isfinite(df["btm_glcm_contrast_7"].iloc[0])
        assert np.isfinite(df["btm_glcm_homogeneity_7"].iloc[0])

    def test_rotation_invariance_approx(self):
        """Direction-averaged homogeneity should be similar for transposed array."""
        rng = np.random.default_rng(5)
        arr = rng.random((50, 50)).astype(np.float32)
        rows, cols = _make_points(5, arr.shape)

        df_orig = compute_glcm_texture(arr, rows, cols, scales=[9])
        # Transpose rows→cols mapping to keep same spatial regions
        df_trans = compute_glcm_texture(arr.T, cols, rows, scales=[9])

        mean_orig = df_orig["btm_glcm_homogeneity_9"].mean()
        mean_trans = df_trans["btm_glcm_homogeneity_9"].mean()
        # Allow ±20% relative difference (direction-averaged GLCM is ~rotation-invariant)
        assert abs(mean_orig - mean_trans) < 0.2 * max(
            abs(mean_orig), abs(mean_trans), 0.01
        )
