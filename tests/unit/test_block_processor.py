"""Tests for btm.io.block — BlockProcessor windowed I/O."""

import pathlib

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from btm.io.block import BlockProcessor

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


def _make_test_raster(tmp_path, data: np.ndarray, cell_size: float = 1.0) -> str:
    """Write a small synthetic raster and return its path."""
    path = str(tmp_path / "test.tif")
    h, w = data.shape
    transform = from_bounds(0, 0, w * cell_size, h * cell_size, w, h)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        dtype=str(data.dtype),
        width=w,
        height=h,
        count=1,
        transform=transform,
        crs="EPSG:4326",
        nodata=-9999,
    ) as dst:
        dst.write(data, 1)
    return path


def _identity(array: np.ndarray) -> np.ndarray:
    """Pass-through function for testing."""
    return array.copy()


class TestBlockProcessor:
    @pytest.fixture
    def simple_raster(self, tmp_path):
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        return _make_test_raster(tmp_path, data)

    def test_output_shape_matches_input(self, simple_raster, tmp_path):
        """process() with identity function yields same shape as input."""
        out = str(tmp_path / "out.tif")
        bp = BlockProcessor()
        bp.process(_identity, simple_raster, out, block_size=5, overlap=0)
        with rasterio.open(out) as src:
            arr = src.read(1)
        original = rasterio.open(simple_raster).read(1)
        assert arr.shape == original.shape

    def test_output_values_match_input(self, simple_raster, tmp_path):
        """Identity function: output values equal input values."""
        out = str(tmp_path / "out.tif")
        bp = BlockProcessor()
        bp.process(_identity, simple_raster, out, block_size=5, overlap=0)
        with rasterio.open(simple_raster) as src:
            original = src.read(1)
        with rasterio.open(out) as dst:
            result = dst.read(1)
        np.testing.assert_array_equal(result, original)

    def test_block_size_larger_than_raster(self, simple_raster, tmp_path):
        """block_size > raster dimensions: handled as single tile."""
        out = str(tmp_path / "big.tif")
        bp = BlockProcessor()
        bp.process(_identity, simple_raster, out, block_size=1024, overlap=0)
        with rasterio.open(simple_raster) as src:
            original = src.read(1)
        with rasterio.open(out) as dst:
            result = dst.read(1)
        np.testing.assert_array_equal(result, original)
