"""Unit tests for btm.features.extract — BTM feature extraction for ML."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from btm.features.extract import extract_btm_features, sample_raster_at_points

DATA_DIR = pathlib.Path(__file__).parent.parent / "data"
BATHY_TIF = DATA_DIR / "bathy5m_clip.tif"
CLASSDICT = DATA_DIR / "fagatelebay.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_points(n: int = 4) -> pd.DataFrame:
    """Build a tiny sample-point DataFrame by reading the raster corners."""
    import rasterio

    with rasterio.open(BATHY_TIF) as src:
        bounds = src.bounds
    # place points interior to the raster (avoid edge nodata)
    xs = np.linspace(bounds.left + 10, bounds.right - 10, n)
    ys = np.linspace(bounds.bottom + 10, bounds.top - 10, n)
    return pd.DataFrame({"ID": range(1, n + 1), "x": xs, "y": ys})


# ---------------------------------------------------------------------------
# sample_raster_at_points
# ---------------------------------------------------------------------------


class TestSampleRasterAtPoints:
    def test_samples_correct_value(self):
        import rasterio

        with rasterio.open(BATHY_TIF) as src:
            arr = src.read(1)
            transform = src.transform
            # Sample the exact centre of cell (0,0)
            cx = transform * (0.5, 0.5)  # (col+0.5, row+0.5) -> geographic
            xs = np.array([cx[0]])
            ys = np.array([cx[1]])

        vals = sample_raster_at_points(arr, transform, xs, ys)
        assert vals.shape == (1,)
        assert np.isfinite(vals[0])
        assert float(vals[0]) == pytest.approx(float(arr[0, 0]), abs=1e-3)

    def test_out_of_bounds_returns_nan(self):
        import rasterio

        with rasterio.open(BATHY_TIF) as src:
            arr = src.read(1)
            transform = src.transform
            # Way outside the raster
            xs = np.array([1e9])
            ys = np.array([1e9])
        vals = sample_raster_at_points(arr, transform, xs, ys)
        assert np.isnan(vals[0])

    def test_nodata_becomes_nan(self):
        arr = np.array([[1.0, -9999.0], [3.0, 4.0]])
        from rasterio.transform import from_bounds

        transform = from_bounds(0, 0, 2, 2, 2, 2)
        xs = np.array([0.5, 1.5])
        ys = np.array([1.5, 1.5])
        vals = sample_raster_at_points(arr, transform, xs, ys, nodata=-9999.0)
        assert float(vals[0]) == pytest.approx(1.0)
        assert np.isnan(vals[1])


# ---------------------------------------------------------------------------
# extract_btm_features
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_points():
    return _make_points(6)


class TestExtractBtmFeatures:
    def test_returns_dataframe(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_original_columns(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        for col in ("ID", "x", "y"):
            assert col in result.columns

    def test_same_row_count(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        assert len(result) == len(sample_points)

    def test_btm_derivative_columns_present(self, sample_points):
        expected = [
            "btm_broad_bpi",
            "btm_fine_bpi",
            "btm_broad_std",
            "btm_fine_std",
            "btm_slope",
            "btm_vrm",
            "btm_surface_ratio",
        ]
        result = extract_btm_features(sample_points, BATHY_TIF)
        for col in expected:
            assert col in result.columns, f"Missing expected column: {col}"

    def test_interaction_columns_present(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        for col in ("btm_bpi_magnitude", "btm_broad_x_fine_std", "btm_rough_total"):
            assert col in result.columns

    def test_no_interaction_columns_when_disabled(self, sample_points):
        result = extract_btm_features(
            sample_points, BATHY_TIF, include_interactions=False
        )
        assert "btm_bpi_magnitude" not in result.columns

    def test_slope_values_in_reasonable_range(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        slope = result["btm_slope"].dropna()
        assert (slope >= 0).all(), "Slope must be non-negative"
        assert (slope < 90).all(), "Slope must be < 90°"

    def test_vrm_values_in_unit_interval(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        vrm = result["btm_vrm"].dropna()
        assert (vrm >= 0).all()
        assert (vrm <= 1.0).all()

    def test_surface_ratio_gte_one(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF)
        sr = result["btm_surface_ratio"].dropna()
        assert (sr >= 1.0 - 1e-6).all(), "Surface ratio must be ≥ 1.0 (planar)"

    def test_custom_prefix(self, sample_points):
        result = extract_btm_features(sample_points, BATHY_TIF, btm_prefix="terrain_")
        assert "terrain_slope" in result.columns
        assert "btm_slope" not in result.columns

    def test_include_rule_class_requires_classdict(self, sample_points):
        with pytest.raises(ValueError, match="classification_file"):
            extract_btm_features(sample_points, BATHY_TIF, include_rule_class=True)

    def test_include_rule_class_with_classdict(self, sample_points):
        result = extract_btm_features(
            sample_points,
            BATHY_TIF,
            include_rule_class=True,
            classification_file=str(CLASSDICT),
        )
        assert "btm_rule_class" in result.columns
        codes = result["btm_rule_class"].dropna()
        # Class codes should be integers 0–11 (0 = unclassified)
        assert codes.between(0, 11).all()

    def test_writes_derivative_rasters_when_outdir_given(self, sample_points, tmp_path):
        extract_btm_features(
            sample_points, BATHY_TIF, outdir=tmp_path, include_interactions=False
        )
        written = list(tmp_path.glob("btm_*.tif"))
        assert len(written) >= 7, f"Expected ≥7 rasters, got {len(written)}"

    def test_compatible_with_merge_on_id(self, sample_points):
        """The output can be merged with another per-point DataFrame on ID."""
        btm = extract_btm_features(sample_points, BATHY_TIF)
        other = sample_points[["ID"]].copy()
        other["backscatter"] = np.random.default_rng(0).uniform(
            -30, -10, len(sample_points)
        )
        combined = other.merge(btm.drop(columns=["x", "y"]), on="ID")
        assert len(combined) == len(sample_points)
        assert "btm_slope" in combined.columns
        assert "backscatter" in combined.columns
