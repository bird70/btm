from __future__ import annotations

import numpy as np
import pandas as pd

from benthic_model.features.spatial_context import add_spatial_context_features


def engineer_features(raw_features: pd.DataFrame) -> pd.DataFrame:
    features = raw_features.copy()

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median = features[col].median()
        features[col] = features[col].fillna(median if np.isfinite(median) else 0.0)

    if {"bathymetry", "backscatter"}.issubset(features.columns):
        features["bathymetry_x_backscatter"] = features["bathymetry"] * features["backscatter"]
        features["acoustic_hardness_proxy"] = features["backscatter"] / (
            np.abs(features["bathymetry"]) + 1.0
        )

    bathy_std = features.get("bathymetry_std_3", pd.Series(0.0, index=features.index))
    back_std = features.get("backscatter_std_3", pd.Series(0.0, index=features.index))
    features["relief_index"] = bathy_std + back_std

    features = add_spatial_context_features(features)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def select_model_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded = {"ID", "id", "class", "bathymetry_flag", "backscatter_flag"}
    cols = [c for c in features.columns if c not in excluded]
    return [c for c in cols if np.issubdtype(features[c].dtype, np.number)]
