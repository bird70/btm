from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {"ID", "id", "class", "location_id"}
    cols = [c for c in frame.columns if c not in ignored and pd.api.types.is_numeric_dtype(frame[c])]
    return cols


def run_deeplab_adapter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    class_weights: dict[str, float],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train a tree-based classifier as a practical DeepLabV3+ proxy."""
    cols = _feature_columns(train_df)
    if not cols:
        raise ValueError("No numeric feature columns found for deeplab adapter")

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        class_weight=class_weights,
        min_samples_leaf=2,
    )
    model.fit(train_df[cols].to_numpy(), train_df["class"].to_numpy())

    classes = list(model.classes_)
    val_probs = pd.DataFrame(model.predict_proba(val_df[cols].to_numpy()), columns=[f"prob_{c}" for c in classes])
    test_probs = pd.DataFrame(model.predict_proba(test_df[cols].to_numpy()), columns=[f"prob_{c}" for c in classes])
    return val_probs, test_probs, classes
