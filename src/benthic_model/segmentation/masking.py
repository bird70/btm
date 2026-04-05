from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from benthic_model.segmentation.grid_mapping import attach_grid_indices, infer_grid_shape


def build_centered_window_mask(
    frame: pd.DataFrame,
    *,
    class_col: str = "class",
    window_size: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build a 2D class index mask using centered fixed-size windows around labeled points."""
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd")
    if class_col not in frame.columns:
        raise ValueError(f"Expected '{class_col}' column")

    labeled = attach_grid_indices(frame)
    classes = list(OrderedDict.fromkeys(labeled[class_col].astype(str).tolist()))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}

    height, width = infer_grid_shape(labeled)
    mask = np.full((height, width), fill_value=-1, dtype=np.int16)

    half = window_size // 2
    for _, row in labeled.iterrows():
        gx = int(row["grid_x"])
        gy = int(row["grid_y"])
        cls = class_to_idx[str(row[class_col])]

        y_min = max(0, gy - half)
        y_max = min(height, gy + half + 1)
        x_min = max(0, gx - half)
        x_max = min(width, gx + half + 1)

        window = mask[y_min:y_max, x_min:x_max]
        # Deterministic overlap policy: keep earliest assignment, fill only unset cells.
        window[window == -1] = cls

    metadata = {
        "shape": [height, width],
        "window_size": window_size,
        "class_index_map": class_to_idx,
        "expansion_rule": "centered_5x5" if window_size == 5 else f"centered_{window_size}x{window_size}",
        "overlap_policy": "deterministic_priority",
    }
    return mask, metadata
