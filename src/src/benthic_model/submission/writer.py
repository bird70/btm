from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def validate_submission_predictions(
    predictions: pd.DataFrame,
    sample_submission: pd.DataFrame,
    allowed_classes: set[str],
) -> None:
    required_cols = ["ID", "class"]
    if list(predictions.columns) != required_cols:
        raise ValueError("Predictions must have exact columns: ID,class")

    if predictions["ID"].duplicated().any():
        raise ValueError("Duplicate IDs found in predictions.")

    expected_ids = set(sample_submission["ID"].tolist())
    predicted_ids = set(predictions["ID"].tolist())

    missing = expected_ids - predicted_ids
    extra = predicted_ids - expected_ids

    if missing:
        raise ValueError(f"Missing prediction IDs: {sorted(missing)[:5]}")
    if extra:
        raise ValueError(f"Unexpected prediction IDs: {sorted(extra)[:5]}")

    unknown_classes = sorted(set(predictions["class"].tolist()) - allowed_classes)
    if unknown_classes:
        raise ValueError(f"Predictions contain unknown classes: {unknown_classes}")


def write_submission(
    predictions_path: str | Path,
    sample_submission_path: str | Path,
    output_path: str | Path,
    allowed_classes: set[str],
) -> Path:
    pred = pd.read_csv(predictions_path)
    sample = pd.read_csv(sample_submission_path)
    validate_submission_predictions(pred, sample, allowed_classes)

    out = pred[["ID", "class"]].copy()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def load_allowed_classes(
    path: str | Path = "artifacts/experiments/class_vocabulary.json",
) -> set[str]:
    vocab_path = Path(path)
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Class vocabulary file not found: {vocab_path}. Run training first to establish labels."
        )

    classes = json.loads(vocab_path.read_text(encoding="utf-8"))
    return set(str(v) for v in classes)
