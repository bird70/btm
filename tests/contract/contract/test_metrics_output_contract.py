import json
from pathlib import Path

import pandas as pd

from benthic_model.evaluation.compare import write_machine_readable_metric_summary


def test_machine_readable_metrics_schema(tmp_path: Path) -> None:
    out_json = tmp_path / "metrics.json"
    out_csv = tmp_path / "metrics.csv"

    payload = {
        "run_id": "run-123",
        "run_type": "candidate",
        "weighted_f1": 0.81,
        "per_class_f1": {"ALG": 0.79, "NVB": 0.83},
        "comparison": {"delta_weighted_f1": 0.01},
    }

    write_machine_readable_metric_summary(payload, out_json, out_csv)

    parsed = json.loads(out_json.read_text(encoding="utf-8"))
    frame = pd.read_csv(out_csv)

    assert parsed["run_id"] == "run-123"
    assert "weighted_f1" in parsed
    assert "class" in frame.columns
    assert "f1" in frame.columns
