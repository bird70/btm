"""Unit tests for btm.core.multiscale — multi-scale terrain feature computation."""

from __future__ import annotations

import numpy as np
import pytest

from btm.core.multiscale import (
    compute_rdmv,
    focal_mean_multiscale,
    multiscale_column_names,
)


class TestMultiscaleColumnNames:
    def test_basic(self):
        names = multiscale_column_names("slope", [3, 7, 11])
        assert names == ["btm_slope_3", "btm_slope_7", "btm_slope_11"]

    def test_single_scale(self):
        assert multiscale_column_names("vrm", [21]) == ["btm_vrm_21"]

    def test_rdmv_naming(self):
        assert multiscale_column_names("rdmv", [3, 7]) == ["btm_rdmv_3", "btm_rdmv_7"]


class TestFocalMeanMultiscale:
    def _make_array(self, rows: int = 20, cols: int = 20, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random((rows, cols)).astype(np.float32)

    def test_output_shape(self):
        arr = self._make_array()
        result = focal_mean_multiscale("slope", arr, [3, 7])
        assert set(result.keys()) == {3, 7}
        assert result[3].shape == arr.shape
        assert result[7].shape == arr.shape

    def test_output_dtype_float64(self):
        arr = self._make_array()
        result = focal_mean_multiscale("slope", arr, [3])
        assert result[3].dtype == np.float64

    def test_smoothing_larger_scale_lower_std(self):
        """Larger scale → greater smoothing → lower standard deviation."""
        rng = np.random.default_rng(42)
        noisy = rng.standard_normal((50, 50)).astype(np.float32)
        result = focal_mean_multiscale("slope", noisy, [3, 21])
        assert result[21].std() < result[3].std()

    def test_boundary_reflect_no_nan(self):
        """Corner cells must not return NaN when mode='reflect'."""
        arr = self._make_array(rows=10, cols=10)
        result = focal_mean_multiscale("slope", arr, [7])
        corner_val = result[7][0, 0]
        assert np.isfinite(
            corner_val
        ), f"Corner value should be finite, got {corner_val}"

    def test_uniform_array_identity(self):
        """Focal mean of a uniform array at any scale equals the constant."""
        arr = np.full((15, 15), 3.14, dtype=np.float32)
        result = focal_mean_multiscale("surface_ratio", arr, [3, 11])
        assert result[3] == pytest.approx(3.14, abs=1e-6)
        assert result[11] == pytest.approx(3.14, abs=1e-6)


class TestComputeRdmv:
    def test_flat_array_returns_zeros(self):
        """Flat surface has zero RDMV at every scale (std=0 case)."""
        arr = np.full((5, 5), -20.0, dtype=np.float64)
        rdmv = compute_rdmv(arr, scale=3)
        assert rdmv.shape == arr.shape
        np.testing.assert_array_equal(rdmv, 0.0)

    def test_nonflat_ridge_positive_depression_negative(self):
        """Ridge peak should have positive RDMV; depression trough negative."""
        depth = np.full((21, 21), -10.0, dtype=np.float64)
        # Place a ridge in the centre
        depth[10, 10] = -5.0  # shallower → positive RDMV
        # Place a depression offset
        depth[5, 5] = -20.0  # deeper → negative RDMV

        rdmv = compute_rdmv(depth, scale=7)
        assert rdmv[10, 10] > 0, "Ridge should have positive RDMV"
        assert rdmv[5, 5] < 0, "Depression should have negative RDMV"

    def test_output_shape(self):
        arr = np.random.default_rng(0).standard_normal((30, 25)).astype(np.float64)
        rdmv = compute_rdmv(arr, scale=5)
        assert rdmv.shape == (30, 25)

    def test_output_dtype_float64(self):
        arr = np.ones((10, 10), dtype=np.float32)
        rdmv = compute_rdmv(arr, scale=3)
        assert rdmv.dtype == np.float64

    def test_no_nan_in_output(self):
        """No NaN should appear in RDMV output for a well-behaved array."""
        arr = np.random.default_rng(1).standard_normal((20, 20)).astype(np.float64)
        rdmv = compute_rdmv(arr, scale=5)
        assert not np.any(np.isnan(rdmv))
