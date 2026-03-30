from __future__ import annotations

from pathlib import Path

from benthic_model.experiment.registry import find_run, get_run_lineage


def compare_runs_with_tolerance(
    run_id_a: str,
    run_id_b: str,
    tolerance: float = 0.01,
    registry_path: str | Path = "artifacts/experiments/run_registry.jsonl",
) -> dict:
    run_a = find_run(run_id_a, registry_path)
    run_b = find_run(run_id_b, registry_path)

    if run_a is None or run_b is None:
        missing = run_id_a if run_a is None else run_id_b
        raise ValueError(f"Run ID not found in registry: {missing}")

    f1_a = float(run_a.get("metric_weighted_f1") or 0.0)
    f1_b = float(run_b.get("metric_weighted_f1") or 0.0)
    delta = abs(f1_a - f1_b)

    return {
        "run_id_a": run_id_a,
        "run_id_b": run_id_b,
        "weighted_f1_a": f1_a,
        "weighted_f1_b": f1_b,
        "delta_weighted_f1": delta,
        "tolerance": tolerance,
        "within_tolerance": delta <= tolerance,
        "lineage_a": get_run_lineage(run_id_a, registry_path),
        "lineage_b": get_run_lineage(run_id_b, registry_path),
    }


def summarize_drift(run_id: str, tolerance: float = 0.01) -> dict:
    lineage = get_run_lineage(run_id)
    if not lineage:
        return {
            "run_id": run_id,
            "status": "no_lineage",
            "drift_detected": False,
            "details": [],
        }

    current = find_run(run_id)
    if current is None:
        raise ValueError(f"Run ID not found: {run_id}")

    current_f1 = float(current.get("metric_weighted_f1") or 0.0)
    details = []
    drift_detected = False

    for parent in lineage:
        parent_f1 = float(parent.get("metric_weighted_f1") or 0.0)
        delta = abs(current_f1 - parent_f1)
        details.append(
            {
                "reference_run_id": parent.get("run_id"),
                "reference_weighted_f1": parent_f1,
                "delta_weighted_f1": delta,
                "within_tolerance": delta <= tolerance,
            }
        )
        if delta > tolerance:
            drift_detected = True

    return {
        "run_id": run_id,
        "status": "evaluated",
        "drift_detected": drift_detected,
        "tolerance": tolerance,
        "details": details,
    }
