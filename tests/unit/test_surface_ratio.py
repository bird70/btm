"""Failing tests for btm.core.surface_ratio — compute_surface_planar_ratio (TDD)."""

import numpy as np

from btm.core.surface_ratio import compute_surface_planar_ratio


class TestComputeSurfaceRatio:
    def test_output_dtype_float32(self):
        arr = np.zeros((20, 20), dtype=np.float64)
        result = compute_surface_planar_ratio(arr, cell_size=5.0)
        assert result.dtype == np.float32

    def test_output_shape(self):
        arr = np.zeros((15, 20), dtype=np.float32)
        result = compute_surface_planar_ratio(arr, cell_size=5.0)
        assert result.shape == arr.shape

    def test_flat_surface_ratio_is_one(self):
        """Flat raster → surface area == planar area → ratio ≈ 1.0."""
        arr = np.zeros((20, 20), dtype=np.float64)
        result = compute_surface_planar_ratio(arr, cell_size=5.0)
        interior = result[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 1.0, atol=1e-4)

    def test_sloped_surface_ratio_gt_one(self):
        """Sloped surface has greater surface area than planar area."""
        size = 20
        cell_size = 1.0
        col = np.arange(size, dtype=np.float64)
        arr = np.tile(col, (size, 1))
        result = compute_surface_planar_ratio(arr, cell_size=cell_size)
        interior = result[1:-1, 1:-1]
        assert np.all(interior > 1.0)

    def test_cell_size_used_correctly(self):
        """Different cell_size values should produce different ratios for same slope."""
        size = 20
        col = np.arange(size, dtype=np.float64)
        arr = np.tile(col, (size, 1))
        r1 = compute_surface_planar_ratio(arr, cell_size=1.0)
        r2 = compute_surface_planar_ratio(arr, cell_size=10.0)
        # Absolute values differ; centre values should not be equal
        assert not np.allclose(r1[5:-5, 5:-5], r2[5:-5, 5:-5], atol=0.01)
