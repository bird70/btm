from __future__ import annotations

import pandas as pd


def attach_grid_indices(
    frame: pd.DataFrame,
    *,
    x_col: str = "x",
    y_col: str = "y",
) -> pd.DataFrame:
    """Attach integer grid coordinates by rank-ordering x/y values."""
    if x_col not in frame.columns or y_col not in frame.columns:
        raise ValueError(f"Expected '{x_col}' and '{y_col}' columns")

    out = frame.copy()
    out["grid_x"] = out[x_col].rank(method="dense").astype(int) - 1
    out["grid_y"] = out[y_col].rank(method="dense").astype(int) - 1
    return out


def infer_grid_shape(frame: pd.DataFrame) -> tuple[int, int]:
    if "grid_x" not in frame.columns or "grid_y" not in frame.columns:
        raise ValueError("grid_x/grid_y columns are required")
    width = int(frame["grid_x"].max()) + 1
    height = int(frame["grid_y"].max()) + 1
    return height, width
