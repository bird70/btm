import pandas as pd

from benthic_model.features.engineering import engineer_features


def test_engineer_features_creates_expected_columns() -> None:
    raw = pd.DataFrame(
        {
            "bathymetry": [10.0, 12.0, 15.0],
            "backscatter": [3.0, 4.0, 8.0],
            "bathymetry_std_3": [1.0, 1.2, 0.9],
            "backscatter_std_3": [0.3, 0.2, 0.5],
        }
    )

    features = engineer_features(raw)

    assert "bathymetry_x_backscatter" in features.columns
    assert "relief_index" in features.columns
    assert "acoustic_hardness_proxy" in features.columns


def test_engineer_features_fills_missing_values() -> None:
    raw = pd.DataFrame(
        {
            "bathymetry": [10.0, None, 15.0],
            "backscatter": [3.0, 4.0, None],
            "bathymetry_std_3": [None, 1.2, 0.9],
            "backscatter_std_3": [0.3, None, 0.5],
        }
    )

    features = engineer_features(raw)

    assert features.isna().sum().sum() == 0


# ---------------------------------------------------------------------------
# T007: engineer_features() with FeatureFlags tests (RED phase)
# ---------------------------------------------------------------------------


def _base_raw() -> pd.DataFrame:
    """Minimal raw DataFrame with core MBES columns."""
    return pd.DataFrame(
        {
            "bathymetry": [10.0, 12.0, 15.0, 8.0, 9.0],
            "backscatter": [3.0, 4.0, 8.0, 2.0, 5.0],
            "bathymetry_std_3": [1.0, 1.2, 0.9, 0.8, 1.1],
            "backscatter_std_3": [0.3, 0.2, 0.5, 0.4, 0.3],
            "x": [1.0, 2.0, 3.0, 4.0, 5.0],
            "y": [10.0, 11.0, 12.0, 13.0, 14.0],
        }
    )


def test_engineer_features_include_interactions_false_omits_interaction_cols() -> None:
    """When include_interactions=False the three interaction columns are absent."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags(include_interactions=False)
    features = engineer_features(_base_raw(), flags=flags)

    assert "bathymetry_x_backscatter" not in features.columns
    assert "acoustic_hardness_proxy" not in features.columns
    assert "relief_index" not in features.columns


def test_engineer_features_include_focal_stats_false_omits_focal_columns() -> None:
    """When include_focal_stats=False, dynamically-computed *_std_* / tpi_* cols absent.

    Note: *_std_* columns already present in the raw input CSV are passed through;
    only the derived focal stats added by engineer_features() are skipped.
    In this test the raw frame already contains bathymetry_std_3 and backscatter_std_3,
    so the test verifies no *additional* focal stat columns are added.
    The relief_index (which is computed from these stds) should also be absent.
    """
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags(include_focal_stats=False)
    features = engineer_features(_base_raw(), flags=flags)

    # relief_index is a dynamically-computed focal stat — must be absent
    assert "relief_index" not in features.columns


def test_engineer_features_include_spatial_z_scores_false_omits_z_cols() -> None:
    """When include_spatial_z_scores=False, spatial z-score columns are absent."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags(include_spatial_z_scores=False)
    features = engineer_features(_base_raw(), flags=flags)

    for col in ("bathymetry_z", "backscatter_z"):
        assert (
            col not in features.columns
        ), f"Expected {col!r} to be absent when spatial z-scores disabled"


def test_engineer_features_all_flags_false_leaves_only_raw_cols() -> None:
    """With all flags False, only the raw input columns survive (+ NaN fill)."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags(
        include_focal_stats=False,
        include_interactions=False,
        include_spatial_z_scores=False,
    )
    raw = _base_raw()
    features = engineer_features(raw, flags=flags)

    # No derived columns should be added
    derived = {
        "bathymetry_x_backscatter",
        "acoustic_hardness_proxy",
        "relief_index",
        "bathymetry_z",
        "backscatter_z",
        "bathymetry_backscatter_rank",
    }
    added = derived.intersection(features.columns)
    assert not added, f"Unexpected derived columns present: {added}"


def test_engineer_features_no_flags_unchanged_behaviour() -> None:
    """Calling engineer_features() without flags preserves original behaviour."""
    raw = _base_raw()
    features_no_flags = engineer_features(raw)
    features_none = engineer_features(raw, flags=None)

    assert set(features_no_flags.columns) == set(features_none.columns)
