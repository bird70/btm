from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def build_meta_feature_frames(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    val_truth: pd.DataFrame,
    seg_val: pd.DataFrame,
    seg_test: pd.DataFrame,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Create train/test meta-feature frames from RF, MLP and segmentation probabilities."""
    if "class" not in train.columns:
        raise ValueError("train must contain class column")

    feature_cols = [c for c in train.columns if c not in {"class", "ID", "id"}]
    x_all = train[feature_cols].copy()
    y_all = train["class"].copy()

    val_idx = train.index[train.get("ID", train.index + 1).astype(str).isin(val_truth["location_id"].astype(str))]
    train_idx = train.index.difference(val_idx)

    x_train = x_all.loc[train_idx]
    y_train = y_all.loc[train_idx]
    x_val = x_all.loc[val_idx]
    y_val = y_all.loc[val_idx]

    rf = RandomForestClassifier(n_estimators=250, random_state=random_state, min_samples_leaf=2)
    rf.fit(x_train, y_train)

    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=random_state)
    mlp.fit(x_train, y_train)

    classes = list(rf.classes_)
    rf_val = pd.DataFrame(rf.predict_proba(x_val), columns=[f"rf_prob_{c}" for c in classes])
    rf_test = pd.DataFrame(rf.predict_proba(test[feature_cols]), columns=[f"rf_prob_{c}" for c in classes])

    mlp_val = pd.DataFrame(mlp.predict_proba(x_val), columns=[f"mlp_prob_{c}" for c in classes])
    mlp_test = pd.DataFrame(mlp.predict_proba(test[feature_cols]), columns=[f"mlp_prob_{c}" for c in classes])

    seg_prob_cols = [c for c in seg_val.columns if c.startswith("prob_")]
    seg_val_aligned = seg_val.sort_values("location_id").reset_index(drop=True)
    seg_test_aligned = seg_test.sort_values("location_id").reset_index(drop=True)

    y_val_aligned = (
        val_truth.sort_values("location_id")["class"].reset_index(drop=True)
    )

    val_meta = pd.concat(
        [
            rf_val.reset_index(drop=True),
            mlp_val.reset_index(drop=True),
            seg_val_aligned[seg_prob_cols].add_prefix("seg_"),
            seg_val_aligned[["confidence"]].rename(columns={"confidence": "seg_confidence"}),
        ],
        axis=1,
    )

    test_meta = pd.concat(
        [
            rf_test.reset_index(drop=True),
            mlp_test.reset_index(drop=True),
            seg_test_aligned[seg_prob_cols].add_prefix("seg_"),
            seg_test_aligned[["confidence"]].rename(columns={"confidence": "seg_confidence"}),
        ],
        axis=1,
    )

    if val_meta.isna().any().any() or test_meta.isna().any().any():
        raise ValueError("Meta-feature frames contain NaN values")

    return val_meta, test_meta, y_val_aligned
