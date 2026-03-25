"""Tests for btm.core.depth_statistics — focal statistics (TDD)."""

import numpy as np
import pytest

from btm.core.depth_statistics import compute_focal_stats


class TestFocalStats:
    @pytest.fixture
    def small_arr(self):
        rng = np.random.default_rng(42)
        return rng.uniform(-100, 0, (15, 15))

    def test_mean_output_shape(self, small_arr):
        results = compute_focal_stats(small_arr, n_size=3, stats=["mean"])
        assert results["mean"].shape == small_arr.shape

    def test_mean_flat_surface(self):
        arr = np.full((15, 15), -50.0)
        results = compute_focal_stats(arr, n_size=3, stats=["mean"])
        np.testing.assert_allclose(
            results["mean"][2:-2, 2:-2], -50.0, atol=1e-6
        )

    def test_iqr_synthetic(self):
        """IQR of a constant array should be 0."""
        arr = np.full((15, 15), 10.0)
        results = compute_focal_stats(arr, n_size=3, stats=["iqr"])
        inner = results["iqr"][1:-1, 1:-1]
        np.testing.assert_allclose(inner, 0.0, atol=1e-6)

    def test_kurtosis_output_shape(self, small_arr):
        results = compute_focal_stats(small_arr, n_size=3, stats=["kurtosis"])
        assert results["kurtosis"].shape == small_arr.shape

    def test_unknown_stat_raises(self, small_arr):
        with pytest.raises(ValueError, match="Unknown stat"):
            compute_focal_stats(small_arr, n_size=3, stats=["bogus"])

    def test_std_and_variance_consistent(self, small_arr):
        results = compute_focal_stats(small_arr, n_size=3, stats=["std", "variance"])
        np.testing.assert_allclose(
            results["std"][3:-3, 3:-3] ** 2,
            results["variance"][3:-3, 3:-3],
            rtol=1e-5,
        )
