from __future__ import annotations

import csv
import json
from pathlib import Path

from benthic_model.experiment.registry import find_run, load_registry


def enforce_weighted_f1_threshold(
    baseline_weighted_f1: float,
    candidate_weighted_f1: float,
    threshold: float = 0.005,
    override: bool = False,
) -> None:
    degradation = baseline_weighted_f1 - candidate_weighted_f1
    if degradation > threshold and not override:
        raise ValueError(
            "Candidate weighted_f1 degraded beyond threshold "
            f"({degradation:.6f} > {threshold:.6f}) without override evidence."
        )


def _latest_baseline_run(before_run_id: str | None = None) -> dict | None:
    rows = [row for row in load_registry() if row.get("run_type") == "baseline"]
    if not rows:
        return None
    return rows[-1]


def compare_run_to_baseline(
    candidate_run_id: str, fold_scheme: str, override: bool = False
) -> dict:
    candidate = find_run(candidate_run_id)
    if candidate is None:
        raise ValueError(f"Run ID not found: {candidate_run_id}")

    baseline = _latest_baseline_run(before_run_id=candidate_run_id)
    candidate_weighted = float(candidate.get("metric_weighted_f1") or 0.0)
    candidate_per_class = candidate.get("metric_per_class_f1") or {}

    baseline_weighted = (
        float(baseline.get("metric_weighted_f1") or 0.0) if baseline else candidate_weighted
    )
    baseline_per_class = (baseline.get("metric_per_class_f1") or {}) if baseline else {}

    enforce_weighted_f1_threshold(
        baseline_weighted_f1=baseline_weighted,
        candidate_weighted_f1=candidate_weighted,
        threshold=0.005,
        override=override,
    )

    delta = candidate_weighted - baseline_weighted
    return {
        "run_id": candidate_run_id,
        "run_type": candidate.get("run_type", "candidate"),
        "fold_scheme": fold_scheme,
        "weighted_f1": candidate_weighted,
        "per_class_f1": candidate_per_class,
        "comparison": {
            "baseline_run_id": baseline.get("run_id") if baseline else None,
            "baseline_weighted_f1": baseline_weighted,
            "delta_weighted_f1": delta,
            "baseline_per_class_f1": baseline_per_class,
        },
    }


def write_comparison_report(payload: dict, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f"# Model Comparison: {payload['run_id']}",
        "",
        f"- run_type: {payload['run_type']}",
        f"- fold_scheme: {payload['fold_scheme']}",
        f"- weighted_f1: {payload['weighted_f1']:.6f}",
        f"- baseline_run_id: {payload['comparison']['baseline_run_id']}",
        f"- baseline_weighted_f1: {payload['comparison']['baseline_weighted_f1']:.6f}",
        f"- delta_weighted_f1: {payload['comparison']['delta_weighted_f1']:.6f}",
        "",
        "## Per-Class F1",
    ]

    for cls, score in sorted(payload.get("per_class_f1", {}).items()):
        lines.append(f"- {cls}: {float(score):.6f}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_machine_readable_metric_summary(
    payload: dict,
    json_output_path: str | Path,
    csv_output_path: str | Path,
) -> tuple[Path, Path]:
    json_path = Path(json_output_path)
    csv_path = Path(csv_output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "f1"])
        writer.writeheader()
        for cls, score in sorted(payload.get("per_class_f1", {}).items()):
            writer.writerow({"class": cls, "f1": float(score)})

    return json_path, csv_path
