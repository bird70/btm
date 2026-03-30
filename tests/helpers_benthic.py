from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin


def create_mbes_rasters(data_dir: Path, width: int = 40, height: int = 40) -> tuple[Path, Path]:
    mbes_dir = data_dir / "MBES"
    mbes_dir.mkdir(parents=True, exist_ok=True)

    transform = from_origin(0, 40, 1, 1)
    bathy_path = mbes_dir / "bathymetry.tif"
    backscatter_path = mbes_dir / "backscatter.tif"

    bathy = np.arange(width * height, dtype="float32").reshape(height, width)
    backscatter = (bathy * 0.5 + 10).astype("float32")

    for out_path, layer in ((bathy_path, bathy), (backscatter_path, backscatter)):
        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="float32",
            crs="EPSG:32755",
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(layer, 1)

    return bathy_path, backscatter_path


def create_metadata_file(data_dir: Path) -> Path:
    metadata_path = data_dir / "METADATA.MD"
    metadata_path.write_text(
        "Title: Refuge Cove MBES\n"
        "Description: Competition data\n"
        "Source: Kaggle\n"
        "License: Competition terms\n",
        encoding="utf-8",
    )
    return metadata_path


def create_train_test_csvs(data_dir: Path) -> tuple[Path, Path, Path]:
    train = pd.DataFrame(
        {
            "ID": list(range(1, 21)),
            "x": [2.5 + i for i in range(20)],
            "y": [35.5 - (i % 10) for i in range(20)],
            "class": ["ALG" if i % 2 == 0 else "NVB" for i in range(20)],
        }
    )
    test = pd.DataFrame(
        {
            "ID": list(range(101, 111)),
            "x": [5.5 + i for i in range(10)],
            "y": [30.5 - (i % 5) for i in range(10)],
        }
    )
    sample_submission = pd.DataFrame({"ID": test["ID"], "class": ["ALG"] * len(test)})

    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    sample_submission.to_csv(sample_path, index=False)
    return train_path, test_path, sample_path


def create_basic_config(config_dir: Path, run_type: str = "baseline") -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{run_type}.yaml"
    config_path.write_text(
        "seed: 42\n"
        "cv:\n"
        "  n_splits: 3\n"
        "  fold_scheme: spatial_blocked\n"
        "  random_state: 42\n"
        "  spatial_bins: 3\n",
        encoding="utf-8",
    )
    return config_path


def latest_run_id(registry_path: Path) -> str:
    rows = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[-1]["run_id"]
