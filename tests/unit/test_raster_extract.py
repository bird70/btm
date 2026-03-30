from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

from benthic_model.data.raster_extract import sample_raster_values


def _create_test_raster(path: Path, nodata: float = -9999.0) -> Path:
    arr = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, nodata, 6.0],
            [7.0, 8.0, 9.0],
        ],
        dtype="float32",
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=3,
        height=3,
        count=1,
        dtype="float32",
        crs="EPSG:32755",
        transform=from_origin(0, 3, 1, 1),
        nodata=nodata,
    ) as dst:
        dst.write(arr, 1)
    return path


def test_sample_raster_values_marks_out_of_bounds(tmp_path: Path) -> None:
    raster_path = _create_test_raster(tmp_path / "tiny.tif")
    points = pd.DataFrame({"x": [0.5, 100.0], "y": [2.5, 100.0]})

    sampled = sample_raster_values(points, raster_path, prefix="bathymetry")

    assert sampled.loc[0, "bathymetry"] == 1.0
    assert sampled.loc[1, "bathymetry_flag"] == "out_of_bounds"
    assert np.isnan(sampled.loc[1, "bathymetry"])


def test_sample_raster_values_marks_nodata(tmp_path: Path) -> None:
    raster_path = _create_test_raster(tmp_path / "tiny_nodata.tif")
    points = pd.DataFrame({"x": [1.5], "y": [1.5]})

    sampled = sample_raster_values(points, raster_path, prefix="backscatter")

    assert sampled.loc[0, "backscatter_flag"] == "nodata"
    assert np.isnan(sampled.loc[0, "backscatter"])
