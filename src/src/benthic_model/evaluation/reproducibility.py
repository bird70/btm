from __future__ import annotations

import json
from pathlib import Path

from benthic_model.experiment.compare_runs import summarize_drift


def write_reproducibility_report(run_id: str, tolerance: float = 0.01) -> tuple[Path, Path, dict]:
    payload = summarize_drift(run_id, tolerance=tolerance)

    out_dir = Path("reports") / "reproducibility"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{run_id}_reproducibility.json"
    md_path = out_dir / f"{run_id}_reproducibility.md"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        f"# Reproducibility Report: {run_id}",
        "",
        f"- status: {payload.get('status')}",
        f"- drift_detected: {payload.get('drift_detected')}",
        f"- tolerance: {payload.get('tolerance', tolerance)}",
        "",
        "## Drift Details",
    ]

    details = payload.get("details", [])
    if not details:
        lines.append("- No lineage runs available for drift comparison.")
    else:
        for row in details:
            lines.append(
                "- "
                f"reference_run_id={row['reference_run_id']}, "
                f"reference_weighted_f1={row['reference_weighted_f1']:.6f}, "
                f"delta_weighted_f1={row['delta_weighted_f1']:.6f}, "
                f"within_tolerance={row['within_tolerance']}"
            )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path, payload
