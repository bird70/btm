from __future__ import annotations

import numpy as np
import pandas as pd


def add_spatial_context_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    if "bathymetry" in out.columns:
        mean_bathy = float(out["bathymetry"].mean())
        std_bathy = float(out["bathymetry"].std() or 1.0)
        out["bathymetry_z"] = (out["bathymetry"] - mean_bathy) / std_bathy

    if "backscatter" in out.columns:
        mean_back = float(out["backscatter"].mean())
        std_back = float(out["backscatter"].std() or 1.0)
        out["backscatter_z"] = (out["backscatter"] - mean_back) / std_back

    if {"bathymetry", "backscatter"}.issubset(out.columns):
        out["bathymetry_backscatter_rank"] = (
            out["bathymetry"].rank(method="average") + out["backscatter"].rank(method="average")
        ) / (2.0 * len(out))

    out = out.replace([np.inf, -np.inf], np.nan)
    return out
