from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from benthic_model.data.raster_extract import extract_mbes_features
from benthic_model.data.validation import validate_coordinates
from benthic_model.experiment.registry import find_run
from benthic_model.features.engineering import engineer_features


def predict_with_run(
    run_id: str,
    test_csv: str | Path,
    bathymetry_tif: str | Path,
    backscatter_tif: str | Path,
) -> Path:
    run = find_run(run_id)
    if run is None:
        raise ValueError(f"Run ID not found: {run_id}")

    artifacts = run.get("artifacts") or {}
    model_path = Path(artifacts["model"])
    feature_cols_path = Path(artifacts["feature_columns"])

    model = joblib.load(model_path)
    feature_cols = json.loads(feature_cols_path.read_text(encoding="utf-8"))

    test_frame = pd.read_csv(test_csv)
    validate_coordinates(test_frame, x_col="x", y_col="y")

    # Include any pre-computed BTM columns from a BTM-augmented test CSV
    btm_passthrough = [c for c in test_frame.columns if c.startswith("btm_")]
    cols_to_pass = [
        c for c in ["ID", "x", "y"] + btm_passthrough if c in test_frame.columns
    ]
    sampled = extract_mbes_features(
        test_frame[cols_to_pass], bathymetry_tif, backscatter_tif
    )
    engineered = engineer_features(sampled)

    for col in feature_cols:
        if col not in engineered.columns:
            engineered[col] = 0.0

    X = engineered[feature_cols]
    preds = model.predict(X)

    output = pd.DataFrame({"ID": test_frame["ID"], "class": preds})
    output_path = Path("artifacts") / "predictions" / f"{run_id}_test_predictions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return output_path
