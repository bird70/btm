from __future__ import annotations

import pandas as pd


def validate_coordinates(
    frame: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
) -> None:
    """Raise ValueError if required coordinate columns are missing or all-NaN."""
    for col in (x_col, y_col):
        if col not in frame.columns:
            raise ValueError(
                f"Required coordinate column {col!r} not found in DataFrame. "
                f"Available columns: {list(frame.columns)}"
            )
        if frame[col].isna().all():
            raise ValueError(f"Coordinate column {col!r} contains only NaN values.")
