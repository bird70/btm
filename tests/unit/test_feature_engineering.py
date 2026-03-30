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
