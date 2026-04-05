from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    ignored = {"ID", "id", "class", "location_id"}
    cols = [c for c in frame.columns if c not in ignored and pd.api.types.is_numeric_dtype(frame[c])]
    return cols


def run_segformer_adapter(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    class_weights: dict[str, float],
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train a lightweight multinomial classifier as a practical SegFormer proxy."""
    cols = _feature_columns(train_df)
    if not cols:
        raise ValueError("No numeric feature columns found for segformer adapter")

    x_train = train_df[cols].to_numpy()
    x_val = val_df[cols].to_numpy()
    x_test = test_df[cols].to_numpy()

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    model = LogisticRegression(
        max_iter=600,
        solver="lbfgs",
        class_weight=class_weights,
        random_state=random_state,
    )
    model.fit(x_train, train_df["class"].to_numpy())

    classes = list(model.classes_)
    val_probs = pd.DataFrame(model.predict_proba(x_val), columns=[f"prob_{c}" for c in classes])
    test_probs = pd.DataFrame(model.predict_proba(x_test), columns=[f"prob_{c}" for c in classes])
    return val_probs, test_probs, classes
