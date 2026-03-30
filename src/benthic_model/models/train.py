from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
import yaml

from benthic_model.config import PipelineConfig
from benthic_model.data.raster_extract import extract_mbes_features
from benthic_model.data.validation import validate_coordinates
from benthic_model.evaluation.cv import (
    iter_spatial_blocked_folds,
    iter_stratified_random_folds,
)
from benthic_model.evaluation.metrics import per_class_f1, weighted_f1
from benthic_model.experiment.metadata import ExperimentMetadata
from benthic_model.experiment.registry import append_run_metadata
from benthic_model.features.engineering import (
    engineer_features,
    select_model_feature_columns,
)
from benthic_model.models.baseline import build_baseline_model
from benthic_model.models.candidate import (
    build_candidate_model,
    build_catboost_model,
    build_lgbm_model,
    build_rf_lgbm_ensemble_model,
)


def _git_revision() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        )
        return output.strip()
    except Exception:
        return "unknown"


def _config_hash(config_path: Path) -> str:
    content = config_path.read_bytes()
    return hashlib.sha256(content).hexdigest()[:12]


def _build_model(run_type: str, seed: int, model_type: str | None = None):
    resolved = model_type or ("rf" if run_type == "baseline" else "xgb")
    dispatch = {
        "rf": build_baseline_model,
        "xgb": build_candidate_model,
        "lgbm": build_lgbm_model,
        "catboost": build_catboost_model,
        "rf_lgbm_ensemble": build_rf_lgbm_ensemble_model,
    }
    if resolved not in dispatch:
        raise ValueError(f"Unknown model_type: {resolved!r}")
    return dispatch[resolved](seed=seed)


def _cross_validate(
    run_type: str,
    features: pd.DataFrame,
    labels: pd.Series,
    coords: pd.DataFrame,
    fold_scheme: str,
    n_splits: int,
    seed: int,
    model_type: str | None = None,
) -> tuple[float, dict[str, float]]:
    if fold_scheme == "spatial_blocked":
        fold_iter = iter_spatial_blocked_folds(
            x=coords["x"].to_numpy(),
            y=coords["y"].to_numpy(),
            n_splits=n_splits,
            random_state=seed,
        )
    else:
        fold_iter = iter_stratified_random_folds(
            labels=labels.tolist(), n_splits=n_splits, random_state=seed
        )

    weighted_scores: list[float] = []
    per_class_accumulator: dict[str, list[float]] = {}

    for train_idx, test_idx in fold_iter:
        model = _build_model(run_type, seed, model_type)
        model.fit(features.iloc[train_idx], labels.iloc[train_idx])
        pred = model.predict(features.iloc[test_idx])

        fold_weighted = weighted_f1(labels.iloc[test_idx].tolist(), pred.tolist())
        weighted_scores.append(fold_weighted)

        fold_per_class = per_class_f1(labels.iloc[test_idx].tolist(), pred.tolist())
        for cls, score in fold_per_class.items():
            per_class_accumulator.setdefault(cls, []).append(score)

    if not weighted_scores:
        raise ValueError("No folds produced during cross-validation.")

    per_class_mean = {
        cls: float(sum(scores) / len(scores))
        for cls, scores in sorted(per_class_accumulator.items())
    }
    return float(sum(weighted_scores) / len(weighted_scores)), per_class_mean


def train_and_register_run(
    train_csv: str | Path,
    bathymetry_tif: str | Path,
    backscatter_tif: str | Path,
    config_path: str | Path,
    run_type: str,
    seed: int,
    command: str,
) -> ExperimentMetadata:
    train_path = Path(train_csv)
    config_path = Path(config_path)
    frame = pd.read_csv(train_path)

    if "class" not in frame.columns:
        raise ValueError("Training CSV must include a 'class' column.")

    validate_coordinates(frame, x_col="x", y_col="y")
    # Include any pre-computed BTM columns from a BTM-augmented CSV
    btm_passthrough = [c for c in frame.columns if c.startswith("btm_")]
    passthrough_cols = [col for col in ["x", "y"] + btm_passthrough if col in frame.columns]
    sample_points = frame[passthrough_cols].copy()
    sample_points.insert(
        0, "ID", frame["ID"] if "ID" in frame.columns else range(1, len(frame) + 1)
    )

    sampled = extract_mbes_features(sample_points, bathymetry_tif, backscatter_tif)

    # Load structured config for feature_flags and model_type
    pipeline_cfg = PipelineConfig.from_yaml(config_path)
    engineered = engineer_features(sampled, flags=pipeline_cfg.feature_flags)
    feature_cols = select_model_feature_columns(engineered)

    X = engineered[feature_cols]
    y = frame["class"].astype(str)

    with config_path.open("r", encoding="utf-8") as handle:
        config_data = yaml.safe_load(handle) or {}

    cv_cfg = config_data.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 5))
    fold_scheme = str(cv_cfg.get("fold_scheme", "spatial_blocked"))
    model_type = pipeline_cfg.model_type

    weighted, per_class = _cross_validate(
        run_type=run_type,
        features=X,
        labels=y,
        coords=frame[["x", "y"]],
        fold_scheme=fold_scheme,
        n_splits=n_splits,
        seed=seed,
        model_type=model_type,
    )

    final_model = _build_model(run_type, seed, model_type)
    final_model.fit(X, y)

    run_id = f"{run_type}-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
    run_dir = Path("artifacts") / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    joblib.dump(final_model, model_path)

    feature_cols_path = run_dir / "feature_columns.json"
    feature_cols_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    class_vocab = sorted(y.unique().tolist())
    class_vocab_path = run_dir / "class_vocab.json"
    class_vocab_path.write_text(json.dumps(class_vocab, indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    metrics_payload = {
        "run_id": run_id,
        "run_type": run_type,
        "weighted_f1": weighted,
        "per_class_f1": per_class,
        "fold_scheme": fold_scheme,
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    provenance_path = run_dir / "provenance.json"
    provenance_payload = {
        "run_id": run_id,
        "code_revision": _git_revision(),
        "config_ref": str(config_path),
        "config_hash": _config_hash(config_path),
        "seed": seed,
        "fold_scheme": fold_scheme,
        "data_split_ref": str(train_path),
        "command": command,
        "artifacts": {
            "model": str(model_path),
            "feature_columns": str(feature_cols_path),
            "class_vocab": str(class_vocab_path),
            "metrics": str(metrics_path),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance_payload, indent=2), encoding="utf-8"
    )

    global_vocab_path = Path("artifacts") / "experiments" / "class_vocabulary.json"
    global_vocab_path.parent.mkdir(parents=True, exist_ok=True)
    global_vocab_path.write_text(json.dumps(class_vocab, indent=2), encoding="utf-8")

    metadata = ExperimentMetadata(
        run_id=run_id,
        run_type=run_type,
        code_revision=provenance_payload["code_revision"],
        config_ref=str(config_path),
        seed=seed,
        fold_scheme=fold_scheme,
        command=command,
        config_hash=provenance_payload["config_hash"],
        data_split_ref=str(train_path),
        artifacts={
            "model": str(model_path),
            "feature_columns": str(feature_cols_path),
            "class_vocab": str(class_vocab_path),
            "metrics": str(metrics_path),
            "provenance": str(provenance_path),
        },
        metric_weighted_f1=weighted,
        metric_per_class_f1=per_class,
        model_type_used=model_type or ("rf" if run_type == "baseline" else "xgb"),
        feature_flags_used=(
            {
                "include_focal_stats": pipeline_cfg.feature_flags.include_focal_stats,
                "include_interactions": pipeline_cfg.feature_flags.include_interactions,
                "include_spatial_z_scores": pipeline_cfg.feature_flags.include_spatial_z_scores,
                "include_btm_features": pipeline_cfg.feature_flags.include_btm_features,
            }
            if pipeline_cfg.feature_flags is not None
            else {}
        ),
    )

    append_run_metadata(metadata)
    return metadata
