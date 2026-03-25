"""Failing tests for btm.core.bpi — compute_bpi (TDD: write before implementation)."""

import numpy as np
import pytest

from btm.core.bpi import compute_bpi


class TestComputeBpi:
    def test_output_shape_matches_input(self):
        arr = np.ones((20, 20), dtype=np.float32) * -50.0
        result = compute_bpi(arr, inner_radius=1, outer_radius=5, cell_size=5.0)
        assert result.shape == arr.shape

    def test_flat_surface_is_zero(self):
        """Flat raster → BPI should be 0 everywhere (bathy == focal mean)."""
        arr = np.full((30, 30), -20.0, dtype=np.float32)
        result = compute_bpi(arr, inner_radius=2, outer_radius=8, cell_size=5.0)
        np.testing.assert_array_equal(result, 0)

    def test_known_annulus_mean(self):
        """A synthetic 11×11 raster with one elevated centre cell."""
        arr = np.zeros((11, 11), dtype=np.float64)
        arr[5, 5] = 100.0  # elevated centre
        result = compute_bpi(arr, inner_radius=1, outer_radius=3, cell_size=1.0)
        # Centre cell: bathy=100, annular mean ≈ 0 → BPI ≈ +100
        assert result[5, 5] > 0

    def test_integer_rounding(self):
        """Result must be int32."""
        arr = np.full((20, 20), -10.5, dtype=np.float64)
        result = compute_bpi(arr, inner_radius=1, outer_radius=4, cell_size=1.0)
        assert result.dtype == np.int32

    def test_inner_ge_outer_raises(self):
        arr = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_bpi(arr, inner_radius=5, outer_radius=5, cell_size=1.0)

    def test_inner_greater_than_outer_raises(self):
        arr = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError):
            compute_bpi(arr, inner_radius=6, outer_radius=3, cell_size=1.0)

    def test_nodata_propagates(self):
        """Cells in the input with nodata should have nodata in the output."""
        nodata = -9999.0
        arr = np.full((20, 20), -30.0, dtype=np.float64)
        arr[10, 10] = nodata
        result = compute_bpi(
            arr, inner_radius=1, outer_radius=4, cell_size=1.0, nodata=nodata
        )
        assert result[10, 10] == nodata or np.isnan(result[10, 10])
