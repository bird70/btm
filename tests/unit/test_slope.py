"""Failing tests for btm.core.slope — compute_slope (TDD)."""

import numpy as np

from btm.core.slope import compute_slope


class TestComputeSlope:
    def test_output_dtype_float32(self):
        arr = np.zeros((10, 10), dtype=np.float64)
        result = compute_slope(arr, cell_size=5.0)
        assert result.dtype == np.float32

    def test_output_shape(self):
        arr = np.zeros((15, 20), dtype=np.float32)
        result = compute_slope(arr, cell_size=5.0)
        assert result.shape == arr.shape

    def test_flat_surface_slope_is_zero(self):
        arr = np.zeros((20, 20), dtype=np.float64)
        result = compute_slope(arr, cell_size=5.0)
        # Interior cells should be exactly 0
        np.testing.assert_allclose(result[2:-2, 2:-2], 0.0, atol=1e-6)

    def test_range_0_to_90(self):
        arr = np.random.rand(20, 20).astype(np.float64) * 100
        result = compute_slope(arr, cell_size=5.0)
        valid = result[np.isfinite(result)]
        assert np.all(valid >= 0.0)
        assert np.all(valid <= 90.0)

    def test_45_degree_ramp(self):
        """Raster with slope = 1 rise / 1 run → slope ≈ 45°."""
        size = 20
        cell_size = 1.0
        # Each column increases by cell_size → dz/dx = 1 → slope=45°
        col = np.arange(size, dtype=np.float64)
        arr = np.tile(col, (size, 1))
        result = compute_slope(arr, cell_size=cell_size)
        # Interior cells only (edge may differ due to boundary padding)
        interior = result[2:-2, 2:-2]
        np.testing.assert_allclose(interior, 45.0, atol=1.0)

    def test_nodata_border_cells(self):
        nodata = -9999.0
        arr = np.zeros((20, 20), dtype=np.float64)
        arr[0, :] = nodata
        result = compute_slope(arr, cell_size=5.0, nodata=nodata)
        assert result is not None  # should not raise
