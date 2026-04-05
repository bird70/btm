from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split

from benthic_model.segmentation.crf import smooth_probability_frame
from benthic_model.segmentation.deeplab_adapter import run_deeplab_adapter
from benthic_model.segmentation.io import ensure_dir, validate_candidate_metrics, validate_prediction_frame, write_json
from benthic_model.segmentation.segformer_adapter import run_segformer_adapter
from benthic_model.segmentation.types import CandidateRunMetrics
from benthic_model.segmentation.weights import inverse_freq_capped

CANDIDATES = ("segformer_ft", "segformer_ft_crf", "deeplabv3_ft")
DEFAULT_CLASS_COLUMNS = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


def _location_id(frame: pd.DataFrame) -> pd.Series:
    if "ID" in frame.columns:
        return frame["ID"].astype(str)
    if "id" in frame.columns:
        return frame["id"].astype(str)
    return pd.Series([str(i + 1) for i in range(len(frame))], index=frame.index)


def _align_prob_columns(probs: pd.DataFrame) -> pd.DataFrame:
    cols = [f"prob_{c}" for c in DEFAULT_CLASS_COLUMNS]
    out = pd.DataFrame(index=probs.index)
    for col in cols:
        out[col] = probs[col] if col in probs.columns else 0.0
    out = out.div(out.sum(axis=1).replace(0, 1), axis=0)
    return out


def _to_prediction_frame(base: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    probs = _align_prob_columns(probs)
    pred_idx = probs.to_numpy().argmax(axis=1)
    classes = [c.replace("prob_", "") for c in probs.columns]
    predicted = [classes[i] for i in pred_idx]
    confidence = probs.max(axis=1)

    frame = pd.DataFrame(
        {
            "location_id": _location_id(base),
            "predicted_class": predicted,
            "confidence": confidence,
        }
    )
    return pd.concat([frame, probs.reset_index(drop=True)], axis=1)


def _candidate_metrics(candidate: str, y_true: pd.Series, pred_frame: pd.DataFrame, runtime_s: float, cap: float) -> CandidateRunMetrics:
    y_pred = pred_frame["predicted_class"]
    weighted = float(f1_score(y_true, y_pred, average="weighted"))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    true_sgam = y_true == "SGAM"
    pred_sgam = y_pred == "SGAM"
    sgam_recall = float(recall_score(true_sgam, pred_sgam, zero_division=0))
    return CandidateRunMetrics(
        candidate_type=candidate,
        status="success",
        weighted_f1=weighted,
        macro_f1=macro,
        sgam_recall=sgam_recall,
        runtime_minutes=runtime_s / 60.0,
        class_weight_strategy="inverse_freq_capped",
        class_weight_cap=cap,
    )


def run_segmentation_benchmark(config_path: str | Path) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    train_csv = Path(config["input"]["train_csv"])
    test_csv = Path(config["input"]["test_csv"])
    base_dir = Path(config["output"]["base_dir"])
    random_state = int(config["split"].get("random_state", 42))
    val_fraction = float(config["split"].get("val_fraction", 0.2))
    cap = float(config["weights"].get("cap", 5.0))

    candidates = tuple(config.get("candidates", CANDIDATES))
    if tuple(candidates) != CANDIDATES:
        raise ValueError(f"Candidates must be exactly {CANDIDATES}")

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    if "class" not in train.columns:
        raise ValueError("train_csv must include 'class' column")

    stratify = train["class"] if train["class"].nunique() > 1 else None
    train_idx, val_idx = train_test_split(
        train.index,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify,
    )
    train_fold = train.loc[train_idx].reset_index(drop=True)
    val_fold = train.loc[val_idx].reset_index(drop=True)

    class_weights = inverse_freq_capped(train_fold["class"].astype(str).tolist(), cap=cap)

    run_id = f"seg-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run_dir = ensure_dir(base_dir / run_id)

    candidate_rows: list[dict[str, object]] = []
    candidate_outputs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}

    for candidate in candidates:
        started = time.perf_counter()
        try:
            if candidate == "segformer_ft":
                val_probs, test_probs, _ = run_segformer_adapter(
                    train_fold,
                    val_fold,
                    test,
                    class_weights=class_weights,
                    random_state=random_state,
                )
            elif candidate == "segformer_ft_crf":
                val_probs, test_probs, _ = run_segformer_adapter(
                    train_fold,
                    val_fold,
                    test,
                    class_weights=class_weights,
                    random_state=random_state,
                )
                val_probs = smooth_probability_frame(val_probs, val_fold[["x", "y"]])
                test_probs = smooth_probability_frame(test_probs, test[["x", "y"]])
            elif candidate == "deeplabv3_ft":
                val_probs, test_probs, _ = run_deeplab_adapter(
                    train_fold,
                    val_fold,
                    test,
                    class_weights=class_weights,
                    random_state=random_state,
                )
            else:
                raise ValueError(f"Unsupported candidate: {candidate}")

            val_pred = _to_prediction_frame(val_fold, val_probs)
            test_pred = _to_prediction_frame(test, test_probs)
            validate_prediction_frame(val_pred)
            validate_prediction_frame(test_pred)

            metrics = _candidate_metrics(
                candidate,
                val_fold["class"],
                val_pred,
                runtime_s=time.perf_counter() - started,
                cap=cap,
            )
            candidate_rows.append(metrics.to_dict())
            candidate_outputs[candidate] = (val_pred, test_pred)
        except Exception as exc:  # pragma: no cover
            candidate_rows.append(
                CandidateRunMetrics(
                    candidate_type=candidate,
                    status="failed",
                    weighted_f1=None,
                    macro_f1=None,
                    sgam_recall=None,
                    runtime_minutes=(time.perf_counter() - started) / 60.0,
                    class_weight_strategy="inverse_freq_capped",
                    class_weight_cap=cap,
                    message=str(exc),
                ).to_dict()
            )

    success_rows = [r for r in candidate_rows if r["status"] == "success"]
    if not success_rows:
        raise RuntimeError("All segmentation candidates failed")

    success_rows.sort(key=lambda r: float(r["weighted_f1"]), reverse=True)
    recommended = success_rows[0]["candidate_type"]
    rec_val, rec_test = candidate_outputs[str(recommended)]

    rec_val.to_csv(run_dir / "val_location_predictions.csv", index=False)
    rec_test.to_csv(run_dir / "test_location_predictions.csv", index=False)
    pd.DataFrame({"location_id": _location_id(val_fold), "class": val_fold["class"]}).to_csv(
        run_dir / "val_truth.csv", index=False
    )

    metrics_payload: dict[str, object] = {
        "run_id": run_id,
        "candidates": candidate_rows,
        "recommended_candidate": recommended,
        "recommended_path": "kaggle_notebook_hf",
        "class_weight_strategy": "inverse_freq_capped",
        "class_weight_cap": cap,
        "split": {
            "val_fraction": val_fraction,
            "random_state": random_state,
        },
    }
    validate_candidate_metrics(metrics_payload)
    write_json(run_dir / "candidate_metrics.json", metrics_payload)
    return run_dir
