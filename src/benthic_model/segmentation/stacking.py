from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score

from benthic_model.segmentation.io import ensure_dir, load_prediction_frame, write_json
from benthic_model.segmentation.stacking_features import build_meta_feature_frames
from benthic_model.segmentation.types import PromotionDecision


def run_hybrid_stacking(
    *,
    baseline_run: str,
    seg_run: str,
    train_csv: str | Path,
    test_csv: str | Path,
    seg_base_dir: str | Path = "artifacts/segmentation",
    output_runs_dir: str | Path = "artifacts/runs",
    output_predictions_dir: str | Path = "artifacts/predictions",
    baseline_predictions_dir: str | Path = "artifacts/predictions",
    random_state: int = 42,
) -> tuple[Path, Path, Path]:
    """Build hybrid stacker artifacts and return metric/decision/prediction paths."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    seg_dir = Path(seg_base_dir) / seg_run
    seg_val = load_prediction_frame(seg_dir / "val_location_predictions.csv")
    seg_test = load_prediction_frame(seg_dir / "test_location_predictions.csv")
    val_truth = pd.read_csv(seg_dir / "val_truth.csv")

    val_meta, test_meta, y_val = build_meta_feature_frames(
        train,
        test,
        val_truth=val_truth,
        seg_val=seg_val,
        seg_test=seg_test,
        random_state=random_state,
    )

    stacker = LogisticRegression(max_iter=700, random_state=random_state)
    stacker.fit(val_meta, y_val)

    hybrid_val_pred = pd.Series(stacker.predict(val_meta))
    hybrid_test_pred = pd.Series(stacker.predict(test_meta))

    baseline_val_pred = seg_val["predicted_class"].astype(str)

    hybrid_weighted_f1 = float(f1_score(y_val, hybrid_val_pred, average="weighted"))
    baseline_weighted_f1 = float(f1_score(y_val, baseline_val_pred, average="weighted"))

    true_sgam = y_val == "SGAM"
    baseline_sgam = baseline_val_pred == "SGAM"
    hybrid_sgam = hybrid_val_pred == "SGAM"
    baseline_sgam_recall = float(recall_score(true_sgam, baseline_sgam, zero_division=0))
    hybrid_sgam_recall = float(recall_score(true_sgam, hybrid_sgam, zero_division=0))

    gate_passed = (hybrid_weighted_f1 - baseline_weighted_f1) >= 0.02 and (
        hybrid_sgam_recall - baseline_sgam_recall
    ) >= 0.0

    decision = PromotionDecision(
        baseline_weighted_f1=baseline_weighted_f1,
        hybrid_weighted_f1=hybrid_weighted_f1,
        baseline_sgam_recall=baseline_sgam_recall,
        hybrid_sgam_recall=hybrid_sgam_recall,
        gate_passed=gate_passed,
        decision_reason=(
            "Hybrid promoted" if gate_passed else "Gate failed; baseline retained"
        ),
    )

    run_id = f"hybrid-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run_dir = ensure_dir(Path(output_runs_dir) / run_id)
    pred_dir = ensure_dir(output_predictions_dir)

    metrics_payload = {
        "run_id": run_id,
        "baseline_run": baseline_run,
        "seg_run": seg_run,
        "hybrid_weighted_f1": hybrid_weighted_f1,
        "baseline_weighted_f1": baseline_weighted_f1,
        "hybrid_sgam_recall": hybrid_sgam_recall,
        "baseline_sgam_recall": baseline_sgam_recall,
        "gate_passed": gate_passed,
    }
    metrics_path = write_json(run_dir / "hybrid_metrics.json", metrics_payload)
    decision_path = write_json(run_dir / "promotion_decision.json", decision.to_dict())

    if "ID" in test.columns:
        ids = test["ID"]
    elif "id" in test.columns:
        ids = test["id"]
    else:
        ids = pd.Series(range(1, len(test) + 1))

    if gate_passed:
        final_pred = hybrid_test_pred
    else:
        baseline_file = Path(baseline_predictions_dir) / f"{baseline_run}_test_predictions.csv"
        if baseline_file.exists():
            baseline_frame = pd.read_csv(baseline_file)
            if "class" in baseline_frame.columns and len(baseline_frame) == len(test):
                final_pred = baseline_frame["class"].astype(str)
            else:  # pragma: no cover
                final_pred = seg_test["predicted_class"]
        else:
            final_pred = seg_test["predicted_class"]
    pred_frame = pd.DataFrame({"ID": ids, "class": final_pred})
    prediction_path = pred_dir / f"{run_id}_test_predictions.csv"
    pred_frame.to_csv(prediction_path, index=False)

    return metrics_path, decision_path, prediction_path
