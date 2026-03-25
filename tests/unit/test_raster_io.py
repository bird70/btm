"""Tests for btm.io.raster — RasterDataset read/write helpers."""

import pathlib

import numpy as np
import pytest
import rasterio

from btm.io.raster import RasterDataset

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


class TestFromFile:
    def test_reads_shape(self, bathy_raster_path):
        ds = RasterDataset.from_file(bathy_raster_path)
        assert ds.array.ndim == 2
        assert ds.array.shape[0] > 0
        assert ds.array.shape[1] > 0

    def test_reads_crs(self, bathy_raster_path):
        ds = RasterDataset.from_file(bathy_raster_path)
        assert ds.crs is not None

    def test_reads_transform(self, bathy_raster_path):
        ds = RasterDataset.from_file(bathy_raster_path)
        assert ds.transform is not None

    def test_cell_width_height_positive(self, bathy_raster_path):
        ds = RasterDataset.from_file(bathy_raster_path)
        assert ds.cell_width > 0
        assert ds.cell_height > 0

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            RasterDataset.from_file("/nonexistent/path/missing.tif")


class TestCellSize:
    def test_cell_size_is_mean(self, bathy_raster_path):
        ds = RasterDataset.from_file(bathy_raster_path)
        expected = (ds.cell_width + ds.cell_height) / 2
        assert ds.cell_size() == pytest.approx(expected)


class TestToFile:
    def test_writes_geotiff(self, bathy_raster_path, tmp_outdir):
        ds = RasterDataset.from_file(bathy_raster_path)
        out = str(pathlib.Path(tmp_outdir) / "out.tif")
        ds.to_file(out)
        with rasterio.open(out) as src:
            assert src.count == 1
            arr = src.read(1)
            assert arr.shape == ds.array.shape

    def test_writes_lzw(self, bathy_raster_path, tmp_outdir):
        ds = RasterDataset.from_file(bathy_raster_path)
        out = str(pathlib.Path(tmp_outdir) / "lzw.tif")
        ds.to_file(out, compress="lzw")
        with rasterio.open(out) as src:
            assert src.profile.get("compress", "").lower() == "lzw"

    def test_overwrites_existing(self, bathy_raster_path, tmp_outdir):
        ds = RasterDataset.from_file(bathy_raster_path)
        out = str(pathlib.Path(tmp_outdir) / "overwrite.tif")
        ds.to_file(out)
        ds.to_file(out)  # second call must not raise

    def test_roundtrip_values(self, bathy_raster_path, tmp_outdir):
        ds = RasterDataset.from_file(bathy_raster_path)
        out = str(pathlib.Path(tmp_outdir) / "rt.tif")
        ds.to_file(out)
        ds2 = RasterDataset.from_file(out)
        np.testing.assert_array_equal(ds.array, ds2.array)
