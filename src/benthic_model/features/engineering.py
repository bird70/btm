from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from benthic_model.features.spatial_context import add_spatial_context_features

if TYPE_CHECKING:
    from benthic_model.config import FeatureFlags


def engineer_features(
    raw_features: pd.DataFrame,
    flags: "FeatureFlags | None" = None,
    y: "pd.Series | None" = None,
) -> pd.DataFrame:
    features = raw_features.copy()

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        median = features[col].median()
        features[col] = features[col].fillna(median if np.isfinite(median) else 0.0)

    include_interactions = flags is None or flags.include_interactions
    if include_interactions and {"bathymetry", "backscatter"}.issubset(
        features.columns
    ):
        features["bathymetry_x_backscatter"] = (
            features["bathymetry"] * features["backscatter"]
        )
        features["acoustic_hardness_proxy"] = features["backscatter"] / (
            np.abs(features["bathymetry"]) + 1.0
        )

    include_focal_stats = flags is None or flags.include_focal_stats
    # relief_index is a derived focal stat that also counts as an interaction term;
    # only add it when both groups are enabled.
    if include_interactions and include_focal_stats:
        bathy_std = features.get(
            "bathymetry_std_3", pd.Series(0.0, index=features.index)
        )
        back_std = features.get(
            "backscatter_std_3", pd.Series(0.0, index=features.index)
        )
        features["relief_index"] = bathy_std + back_std

    include_spatial_z = flags is None or flags.include_spatial_z_scores
    if include_spatial_z:
        features = add_spatial_context_features(features)

    include_eco = flags is not None and flags.include_eco_features
    if include_eco:
        from benthic_model.features.eco_features import EcoFeatureTransformer

        eco = EcoFeatureTransformer()
        eco_out = eco.fit_transform(features, y=y)
        features["btm_depth_zone"] = eco_out["btm_depth_zone"]
        features["btm_sgam_niche"] = eco_out["btm_sgam_niche"]

    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def select_model_feature_columns(
    features: pd.DataFrame,
    *,
    exclude_coords: bool = False,
) -> list[str]:
    excluded = {"ID", "id", "class", "bathymetry_flag", "backscatter_flag"}
    if exclude_coords:
        excluded |= {"x", "y"}
    cols = [c for c in features.columns if c not in excluded]
    return [c for c in cols if np.issubdtype(features[c].dtype, np.number)]
