from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = {
    "location_id",
    "predicted_class",
    "confidence",
    "prob_ALG",
    "prob_FMAT",
    "prob_NVB",
    "prob_SGAM",
    "prob_SGZ",
}


def ensure_dir(path: Path | str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: Path | str, payload: dict[str, object]) -> Path:
    out = Path(path)
    ensure_dir(out.parent)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def write_mask_artifacts(run_dir: Path | str, mask: np.ndarray, metadata: dict[str, object]) -> None:
    run_path = ensure_dir(run_dir)
    np.save(run_path / "masks.npy", mask)
    write_json(run_path / "mask_metadata.json", metadata)


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_PREDICTION_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Prediction frame missing columns: {sorted(missing)}")

    for col in frame.columns:
        if col.startswith("prob_") or col == "confidence":
            if ((frame[col] < 0) | (frame[col] > 1)).any():
                raise ValueError(f"Column {col} has values outside [0,1]")


def validate_candidate_metrics(payload: dict[str, object]) -> None:
    if "candidates" not in payload:
        raise ValueError("candidate_metrics payload missing 'candidates'")
    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) == 0:
        raise ValueError("candidate_metrics payload requires non-empty candidates list")

    required = {"candidate_type", "status", "runtime_minutes"}
    for row in candidates:
        if not isinstance(row, dict):
            raise ValueError("candidate_metrics entries must be dicts")
        missing = required - set(row.keys())
        if missing:
            raise ValueError(f"candidate row missing fields: {sorted(missing)}")


def load_prediction_frame(path: Path | str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    validate_prediction_frame(frame)
    return frame
