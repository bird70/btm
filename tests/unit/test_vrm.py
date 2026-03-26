"""Failing tests for btm.core.vrm — compute_vrm (TDD)."""

import numpy as np
import pytest

from btm.core.vrm import compute_vrm


class TestComputeVrm:
    def test_output_dtype_float32(self):
        arr = np.zeros((20, 20), dtype=np.float64)
        result = compute_vrm(arr, neighborhood_size=3, cell_size=5.0)
        assert result.dtype == np.float32

    def test_output_shape(self):
        arr = np.zeros((25, 25), dtype=np.float32)
        result = compute_vrm(arr, neighborhood_size=3, cell_size=5.0)
        assert result.shape == arr.shape

    def test_flat_surface_vrm_near_zero(self):
        """Perfectly flat surface → VRM ≈ 0 (all vectors align perfectly)."""
        arr = np.zeros((30, 30), dtype=np.float64)
        result = compute_vrm(arr, neighborhood_size=3, cell_size=5.0)
        interior = result[3:-3, 3:-3]
        np.testing.assert_allclose(interior, 0.0, atol=1e-5)

    def test_range_0_to_1(self):
        arr = np.random.rand(20, 20).astype(np.float64) * 100
        result = compute_vrm(arr, neighborhood_size=3, cell_size=5.0)
        valid = result[np.isfinite(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 1.0 + 1e-6)

    def test_neighborhood_size_1_raises(self):
        arr = np.zeros((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_vrm(arr, neighborhood_size=1, cell_size=5.0)

    def test_output_shape_matches_input(self):
        arr = np.random.rand(15, 15).astype(np.float32)
        result = compute_vrm(arr, neighborhood_size=3, cell_size=1.0)
        assert result.shape == arr.shape
