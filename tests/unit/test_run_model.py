"""Failing tests for btm.core.model — run_full_model (TDD, US1)."""

import pathlib

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


class TestRunFullModel:
    def test_returns_6_outputs_with_intermediates(self, bathy_path, classdict_path, tmp_path):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10, broad_bpi_outer=30,
            fine_bpi_inner=1, fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
            keep_intermediates=True,
        )
        assert len(outputs) == 6, f"Expected 6 outputs, got {len(outputs)}: {list(outputs)}"

    def test_returns_1_output_no_intermediates(self, bathy_path, classdict_path, tmp_path):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10, broad_bpi_outer=30,
            fine_bpi_inner=1, fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
            keep_intermediates=False,
        )
        assert len(outputs) == 1

    def test_outputs_are_valid_geotiffs(self, bathy_path, classdict_path, tmp_path):
        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10, broad_bpi_outer=30,
            fine_bpi_inner=1, fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
        )
        for name, path in outputs.items():
            assert pathlib.Path(path).exists(), f"Output missing: {name} → {path}"
            with rasterio.open(path) as src:
                assert src.count == 1, f"{name} must be single-band"
                assert src.width > 0

    def test_output_crs_matches_input(self, bathy_path, classdict_path, tmp_path):
        with rasterio.open(bathy_path) as src:
            expected_crs = src.crs

        outputs = run_full_model(
            bathy_path,
            broad_bpi_inner=10, broad_bpi_outer=30,
            fine_bpi_inner=1, fine_bpi_outer=5,
            classification_file=classdict_path,
            outdir=str(tmp_path),
        )
        for name, path in outputs.items():
            with rasterio.open(path) as dst:
                assert dst.crs == expected_crs, f"CRS mismatch for {name}"

    def test_overwrite_succeeds(self, bathy_path, classdict_path, tmp_path):
        for _ in range(2):
            run_full_model(
                bathy_path,
                broad_bpi_inner=10, broad_bpi_outer=30,
                fine_bpi_inner=1, fine_bpi_outer=5,
                classification_file=classdict_path,
                outdir=str(tmp_path),
            )
