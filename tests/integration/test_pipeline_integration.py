"""Integration tests for the complete BTM pipeline (US1)."""

import pathlib

import numpy as np
import pytest
import rasterio

from btm.core.model import run_full_model

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"


@pytest.fixture
def bathy_path():
    p = DATA_DIR / "bathy5m_clip.tif"
    assert p.exists(), f"Test raster not found: {p}"
    return str(p)


@pytest.fixture
def classdict_path():
    p = DATA_DIR / "fagatelebay.csv"
    assert p.exists(), f"Test CSV not found: {p}"
    return str(p)


class TestPipelineIntegration:
    def test_produces_6_non_empty_outputs(self, bathy_path, classdict_path, tmp_path):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10,
            broad_bpi_outer=30,
            fine_bpi_inner=1,
            fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
        )
        assert len(outputs) == 6
        for name, path in outputs.items():
            assert pathlib.Path(path).stat().st_size > 0, f"Empty output: {name}"

    def test_classified_zones_has_multiple_classes(
        self, bathy_path, classdict_path, tmp_path
    ):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10,
            broad_bpi_outer=30,
            fine_bpi_inner=1,
            fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
        )
        with rasterio.open(outputs["classified_zones"]) as src:
            arr = src.read(1)
        unique_codes = set(np.unique(arr).tolist()) - {0}
        assert len(unique_codes) >= 1, "Classified zones has no class assignments"

    def test_all_outputs_lzw_compressed(self, bathy_path, classdict_path, tmp_path):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10,
            broad_bpi_outer=30,
            fine_bpi_inner=1,
            fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
        )
        for name, path in outputs.items():
            with rasterio.open(path) as src:
                compress = src.profile.get("compress", "").lower()
                assert (
                    compress == "lzw"
                ), f"{name} not LZW compressed (got {compress!r})"
