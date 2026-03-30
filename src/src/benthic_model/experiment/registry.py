from __future__ import annotations

import json
from pathlib import Path

from benthic_model.experiment.metadata import ExperimentMetadata

DEFAULT_REGISTRY_PATH = Path("artifacts/experiments/run_registry.jsonl")


def ensure_registry_file(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> Path:
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def append_run_metadata(
    metadata: ExperimentMetadata, registry_path: str | Path = DEFAULT_REGISTRY_PATH
) -> None:
    path = ensure_registry_file(registry_path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata.to_dict(), sort_keys=True) + "\n")


def load_registry(registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict]:
    path = Path(registry_path)
    if not path.exists():
        return []

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entry = line.strip()
            if not entry:
                continue
            records.append(json.loads(entry))
    return records


def find_run(run_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> dict | None:
    for record in load_registry(registry_path):
        if record.get("run_id") == run_id:
            return record
    return None


def list_runs_by_type(
    run_type: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH
) -> list[dict]:
    return [row for row in load_registry(registry_path) if row.get("run_type") == run_type]


def get_run_lineage(run_id: str, registry_path: str | Path = DEFAULT_REGISTRY_PATH) -> list[dict]:
    target = find_run(run_id, registry_path)
    if target is None:
        return []

    config_ref = target.get("config_ref")
    seed = target.get("seed")
    run_type = target.get("run_type")
    lineage: list[dict] = []
    for row in load_registry(registry_path):
        if row.get("run_id") == run_id:
            continue
        if (
            row.get("config_ref") == config_ref
            and row.get("seed") == seed
            and row.get("run_type") == run_type
        ):
            lineage.append(row)
    return lineage
