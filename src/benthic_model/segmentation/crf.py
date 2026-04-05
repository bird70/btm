from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


def smooth_probability_frame(
    probs: pd.DataFrame,
    coords: pd.DataFrame,
    *,
    n_neighbors: int = 5,
    blend: float = 0.2,
) -> pd.DataFrame:
    """Apply lightweight neighbor smoothing to probability rows.

    This is a practical stand-in for expensive dense CRF inference in local CLI tests.
    """
    if probs.empty:
        return probs.copy()
    if len(probs) != len(coords):
        raise ValueError("probs and coords must have same row count")

    n = len(probs)
    k = min(max(1, n_neighbors), n)
    model = NearestNeighbors(n_neighbors=k)
    model.fit(coords[["x", "y"]].to_numpy())
    _, indices = model.kneighbors(coords[["x", "y"]].to_numpy())

    arr = probs.to_numpy(dtype=float)
    smoothed = np.empty_like(arr)
    for i in range(n):
        local_mean = arr[indices[i]].mean(axis=0)
        smoothed[i] = (1.0 - blend) * arr[i] + blend * local_mean

    smoothed = np.clip(smoothed, 1e-9, 1.0)
    smoothed = smoothed / smoothed.sum(axis=1, keepdims=True)
    return pd.DataFrame(smoothed, columns=probs.columns, index=probs.index)
