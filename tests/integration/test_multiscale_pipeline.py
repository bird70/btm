"""Integration test for the multi-scale BTM feature extraction pipeline.

Uses committed 50×50 fixture rasters in tests/data/016/ to run
extract_btm_features end-to-end with scales and GLCM enabled.

These rasters were clipped from the centre of the real MBES survey
(data/MBES/bathymetry.tif and backscatter.tif) — all 2500 cells are valid.
Synthetic rasters are NOT used (constitution Principle I).
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
import rasterio

from btm.features.extract import extract_btm_features

DATA_DIR = pathlib.Path(__file__).parent.parent / "data" / "016"
BATHY_TIF = DATA_DIR / "bathymetry.tif"
BACK_TIF = DATA_DIR / "backscatter.tif"


pytestmark = pytest.mark.integration


@pytest.fixture
def sample_points() -> pd.DataFrame:
    """5 evenly-spaced interior points within the 50×50 fixture raster."""
    with rasterio.open(BATHY_TIF) as src:
        bounds = src.bounds
    xs = np.linspace(bounds.left + 1, bounds.right - 1, 5)
    ys = np.linspace(bounds.bottom + 1, bounds.top - 1, 5)
    return pd.DataFrame({"ID": range(1, 6), "x": xs, "y": ys})


class TestMultiscalePipeline:
    def test_multiscale_columns_present(self, sample_points):
        """Output DataFrame must contain btm_{deriv}_{scale} columns for all 7 derivatives."""
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=[3, 7],
            include_glcm=False,
        )
        for deriv in [
            "slope",
            "vrm",
            "surface_ratio",
            "northness",
            "eastness",
            "max_curvature",
            "complexity",
        ]:
            for scale in [3, 7]:
                col = f"btm_{deriv}_{scale}"
                assert col in df.columns, f"Missing column: {col}"

    def test_rdmv_columns_present(self, sample_points):
        """Output DataFrame must contain btm_rdmv_{scale} for each scale."""
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=[3, 7],
            include_glcm=False,
        )
        for scale in [3, 7]:
            col = f"btm_rdmv_{scale}"
            assert col in df.columns, f"Missing column: {col}"

    def test_glcm_columns_present(self, sample_points):
        """Output DataFrame must contain GLCM contrast + homogeneity columns."""
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=[3, 7],
            include_glcm=True,
            backscatter_tif=BACK_TIF,
        )
        for scale in [3, 7]:
            assert f"btm_glcm_contrast_{scale}" in df.columns
            assert f"btm_glcm_homogeneity_{scale}" in df.columns

    def test_output_row_count_matches_input(self, sample_points):
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=[3, 7],
            include_glcm=True,
            backscatter_tif=BACK_TIF,
        )
        assert len(df) == len(sample_points)

    def test_no_all_nan_multiscale_columns(self, sample_points):
        """No multi-scale column should be entirely NaN."""
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=[3, 7],
            include_glcm=True,
            backscatter_tif=BACK_TIF,
        )
        ms_cols = [
            c
            for c in df.columns
            if any(
                c.startswith(prefix)
                for prefix in ["btm_slope_", "btm_vrm_", "btm_rdmv_", "btm_glcm_"]
            )
        ]
        assert len(ms_cols) > 0, "No multi-scale columns found"
        for col in ms_cols:
            assert not df[col].isna().all(), f"Column {col} is entirely NaN"

    def test_backward_compat_no_scales(self, sample_points):
        """Calling with scales=None must not produce any btm_*_INT columns."""
        df = extract_btm_features(
            sample_points,
            bathymetry_tif=BATHY_TIF,
            scales=None,
            include_glcm=False,
        )
        ms_cols = [c for c in df.columns if c.startswith("btm_") and c[-1].isdigit()]
        assert len(ms_cols) == 0, f"Unexpected multi-scale columns: {ms_cols}"

    def test_include_glcm_without_backscatter_raises(self, sample_points):
        """include_glcm=True without backscatter_tif must raise ValueError."""
        with pytest.raises(ValueError, match="backscatter_tif"):
            extract_btm_features(
                sample_points,
                bathymetry_tif=BATHY_TIF,
                scales=[3],
                include_glcm=True,
                backscatter_tif=None,
            )
