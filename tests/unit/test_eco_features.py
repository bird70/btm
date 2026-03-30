"""Unit tests for EcoFeatureTransformer (TDD Red phase — T011-T013).

Tests must FAIL before src/benthic_model/features/eco_features.py exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _synthetic_eco_df(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Minimal DataFrame matching EcoFeatureTransformer input contract."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "bathymetry": rng.uniform(-25, -2, n),
            "btm_fine_bpi": rng.normal(0, 50, n),
            "btm_slope": rng.uniform(0, 15, n),
        }
    )


def _sgam_labels(n: int = 40) -> pd.Series:
    """Label series: first 8 SGAM, rest mixed."""
    labels = ["NVB"] * n
    for i in range(8):
        labels[i] = "SGAM"
    return pd.Series(labels)


# ---------------------------------------------------------------------------
# T011: EcoFeatureTransformer.fit() stores expected threshold attributes
# ---------------------------------------------------------------------------


def test_eco_transformer_fit_stores_depth_bin_edges() -> None:
    """fit() must store depth_bin_edges with len == N_DEPTH_BINS + 1 (5 edges for 4 bins)."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)

    assert hasattr(t, "depth_bin_edges_"), "fit() must set depth_bin_edges_"
    assert len(t.depth_bin_edges_) == 5, (
        f"Expected 5 bin edges (4 bins), got {len(t.depth_bin_edges_)}"
    )


def test_eco_transformer_fit_stores_p25_bpi() -> None:
    """fit() must store p25_bpi_ equal to the 25th percentile of btm_fine_bpi."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)

    expected = float(np.percentile(df["btm_fine_bpi"], 25))
    assert hasattr(t, "p25_bpi_"), "fit() must set p25_bpi_"
    assert t.p25_bpi_ == pytest.approx(expected, rel=1e-6)


def test_eco_transformer_fit_stores_p25_slope() -> None:
    """fit() must store p25_slope_ equal to the 25th percentile of btm_slope."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)

    expected = float(np.percentile(df["btm_slope"], 25))
    assert hasattr(t, "p25_slope_"), "fit() must set p25_slope_"
    assert t.p25_slope_ == pytest.approx(expected, rel=1e-6)


def test_eco_transformer_fit_stores_sgam_depth_range() -> None:
    """fit() must store sgam_depth_min_ and sgam_depth_max_ from SGAM-labelled rows."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)

    sgam_depths = df.loc[y == "SGAM", "bathymetry"]
    assert hasattr(t, "sgam_depth_min_"), "fit() must set sgam_depth_min_"
    assert hasattr(t, "sgam_depth_max_"), "fit() must set sgam_depth_max_"
    assert t.sgam_depth_min_ == pytest.approx(float(sgam_depths.min()), rel=1e-6)
    assert t.sgam_depth_max_ == pytest.approx(float(sgam_depths.max()), rel=1e-6)


def test_eco_transformer_fit_stores_sgam_depth_range_no_sgam_rows() -> None:
    """fit() must not crash when no SGAM rows are present; range is [0, 0]."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = pd.Series(["NVB"] * len(df))
    t = EcoFeatureTransformer()
    t.fit(df, y)  # must not raise

    assert t.sgam_depth_min_ == t.sgam_depth_max_


# ---------------------------------------------------------------------------
# T012: EcoFeatureTransformer.transform() adds correct columns
# ---------------------------------------------------------------------------


def test_eco_transformer_transform_adds_depth_zone() -> None:
    """transform() adds btm_depth_zone with values in {1, 2, 3, 4}."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)
    out = t.transform(df)

    assert "btm_depth_zone" in out.columns, "transform() must add btm_depth_zone"
    assert set(out["btm_depth_zone"].unique()).issubset({1, 2, 3, 4}), (
        f"btm_depth_zone must be in {{1,2,3,4}}, got {set(out['btm_depth_zone'].unique())}"
    )


def test_eco_transformer_transform_adds_sgam_niche() -> None:
    """transform() adds btm_sgam_niche with values in {0, 1}."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)
    out = t.transform(df)

    assert "btm_sgam_niche" in out.columns, "transform() must add btm_sgam_niche"
    assert set(out["btm_sgam_niche"].unique()).issubset({0, 1}), (
        f"btm_sgam_niche must be in {{0,1}}, got {set(out['btm_sgam_niche'].unique())}"
    )


def test_eco_transformer_transform_does_not_mutate_input() -> None:
    """transform() must return a new DataFrame without modifying the input."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    original_cols = set(df.columns)
    t = EcoFeatureTransformer()
    t.fit(df, y)
    t.transform(df)

    assert set(df.columns) == original_cols, "transform() must not modify the input df"


def test_eco_transformer_transform_raises_not_fitted_error() -> None:
    """transform() before fit() must raise sklearn NotFittedError."""
    from sklearn.exceptions import NotFittedError

    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    t = EcoFeatureTransformer()
    with pytest.raises(NotFittedError):
        t.transform(df)


def test_eco_transformer_depth_zone_covers_all_bins() -> None:
    """With sufficient variation in depth, all 4 zone values should appear."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    # Use a large range to ensure all 4 quantile bins are populated
    rng = np.random.default_rng(99)
    df = pd.DataFrame(
        {
            "bathymetry": np.linspace(-100, -1, 100),
            "btm_fine_bpi": rng.normal(0, 50, 100),
            "btm_slope": rng.uniform(0, 15, 100),
        }
    )
    y = pd.Series(["SGAM"] * 5 + ["NVB"] * 95)
    t = EcoFeatureTransformer()
    t.fit(df, y)
    out = t.transform(df)

    assert set(out["btm_depth_zone"].unique()) == {1, 2, 3, 4}, (
        f"Expected all four zones, got {set(out['btm_depth_zone'].unique())}"
    )


# ---------------------------------------------------------------------------
# T013: to_dict() / from_dict() round-trip
# ---------------------------------------------------------------------------


def test_eco_transformer_to_dict_round_trip_transform_identical() -> None:
    """Serialise → reconstruct → transform must produce identical output."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()

    t1 = EcoFeatureTransformer()
    t1.fit(df, y)
    out1 = t1.transform(df)

    state = t1.to_dict()
    t2 = EcoFeatureTransformer.from_dict(state)
    out2 = t2.transform(df)

    pd.testing.assert_frame_equal(out1, out2)


def test_eco_transformer_to_dict_contains_required_keys() -> None:
    """to_dict() output must contain depth_bin_edges, p25_bpi, p25_slope, sgam_depth_min, sgam_depth_max."""
    from benthic_model.features.eco_features import EcoFeatureTransformer

    df = _synthetic_eco_df()
    y = _sgam_labels()
    t = EcoFeatureTransformer()
    t.fit(df, y)
    d = t.to_dict()

    required = {"depth_bin_edges", "p25_bpi", "p25_slope", "sgam_depth_min", "sgam_depth_max"}
    assert required.issubset(d.keys()), (
        f"to_dict() missing keys: {required - d.keys()}"
    )


def test_eco_transformer_from_dict_raises_not_fitted_when_empty() -> None:
    """from_dict({}) must return a transformer whose transform() raises NotFittedError."""
    from sklearn.exceptions import NotFittedError

    from benthic_model.features.eco_features import EcoFeatureTransformer

    t = EcoFeatureTransformer.from_dict({})
    df = _synthetic_eco_df()
    with pytest.raises(NotFittedError):
        t.transform(df)
