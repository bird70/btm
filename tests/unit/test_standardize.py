"""Failing tests for btm.core.standardize — standardize_bpi (TDD)."""

import numpy as np
import pytest

from btm.core.standardize import standardize_bpi


class TestStandardizeBpi:
    def test_output_dtype_int32(self):
        arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
        result = standardize_bpi(arr)
        assert result.dtype == np.int32

    def test_formula_known_values(self):
        """Test: stdBPI = round((val − mean) / std × 100)."""
        arr = np.array([0.0, 100.0, 200.0, 300.0, 400.0], dtype=np.float64)
        mean = arr.mean()
        std = arr.std()
        expected = np.round((arr - mean) / std * 100).astype(np.int32)
        result = standardize_bpi(arr.reshape(1, 5))
        np.testing.assert_array_equal(result.flatten(), expected)

    def test_output_shape_preserved(self):
        arr = np.random.randn(15, 15).astype(np.float32)
        result = standardize_bpi(arr)
        assert result.shape == arr.shape

    def test_zero_std_raises(self):
        """All-constant non-nodata array has std=0 → should raise ValueError."""
        arr = np.full((10, 10), 42.0, dtype=np.float64)
        with pytest.raises((ValueError, ZeroDivisionError)):
            standardize_bpi(arr)

    def test_nodata_excluded_from_stats(self):
        nodata = -9999.0
        arr = np.array([[1.0, 2.0, 3.0, nodata]], dtype=np.float64)
        # Should not raise; nodata excluded from mean/std computation
        result = standardize_bpi(arr, nodata=nodata)
        assert result.dtype == np.int32
