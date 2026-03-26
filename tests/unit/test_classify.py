"""Failing tests for btm.core.classify — classify_terrain (TDD)."""

import numpy as np
import pytest

from btm.classification.reader import ClassEntry
from btm.core.classify import classify_terrain


def _make_entry(
    code,
    depth_lower=None,
    depth_upper=None,
    slope_lower=None,
    slope_upper=None,
    broad_lower=None,
    broad_upper=None,
    fine_lower=None,
    fine_upper=None,
):
    return ClassEntry(
        code=code,
        name=f"Class{code}",
        depth_lower=depth_lower,
        depth_upper=depth_upper,
        slope_lower=slope_lower,
        slope_upper=slope_upper,
        broad_bpi_lower=broad_lower,
        broad_bpi_upper=broad_upper,
        fine_bpi_lower=fine_lower,
        fine_bpi_upper=fine_upper,
    )


@pytest.fixture
def shape():
    return (10, 10)


@pytest.fixture
def zeros(shape):
    return np.zeros(shape, dtype=np.float32)


class TestClassifyTerrain:
    def test_output_dtype_int32(self, shape, zeros):
        classes = [_make_entry(1, broad_lower=0.0)]
        result = classify_terrain(zeros, zeros, zeros, zeros, classes)
        assert result.dtype == np.int32

    def test_output_shape(self, shape, zeros):
        classes = [_make_entry(1)]
        result = classify_terrain(zeros, zeros, zeros, zeros, classes)
        assert result.shape == shape

    def test_correct_class_assigned(self, shape):
        broad = np.full(shape, 50.0, dtype=np.float32)
        fine = np.zeros(shape, dtype=np.float32)
        slope = np.zeros(shape, dtype=np.float32)
        bathy = np.zeros(shape, dtype=np.float32)
        classes = [_make_entry(1, broad_lower=0.0)]  # broad_bpi >= 0
        result = classify_terrain(broad, fine, slope, bathy, classes)
        assert np.all(result == 1)

    def test_no_class_match_gives_zero(self, shape):
        broad = np.zeros(shape, dtype=np.float32)
        fine = np.zeros(shape, dtype=np.float32)
        slope = np.zeros(shape, dtype=np.float32)
        bathy = np.zeros(shape, dtype=np.float32)
        # Class requires broad_bpi > 100 — will not match
        classes = [_make_entry(1, broad_lower=101.0)]
        result = classify_terrain(broad, fine, slope, bathy, classes)
        assert np.all(result == 0)

    def test_first_match_wins(self, shape):
        """First matching class takes priority (CON cascade equivalent)."""
        broad = np.full(shape, 50.0, dtype=np.float32)
        fine = np.zeros(shape, dtype=np.float32)
        slope = np.zeros(shape, dtype=np.float32)
        bathy = np.zeros(shape, dtype=np.float32)
        classes = [
            _make_entry(1, broad_lower=0.0),  # matches everything ≥ 0
            _make_entry(2, broad_lower=0.0),  # also matches
        ]
        result = classify_terrain(broad, fine, slope, bathy, classes)
        assert np.all(result == 1)  # first match wins

    def test_nodata_in_any_input_propagates(self, shape):
        nodata = -9999.0
        broad = np.full(shape, nodata, dtype=np.float64)
        fine = np.zeros(shape, dtype=np.float64)
        slope = np.zeros(shape, dtype=np.float64)
        bathy = np.zeros(shape, dtype=np.float64)
        classes = [_make_entry(1)]
        result = classify_terrain(broad, fine, slope, bathy, classes, nodata=nodata)
        # All broad values are nodata → output should be nodata code or 0
        assert result.dtype == np.int32
