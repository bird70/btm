from pathlib import Path

from benthic_model.experiment.metadata import ExperimentMetadata
from benthic_model.experiment.registry import (
    append_run_metadata,
    find_run,
    get_run_lineage,
    list_runs_by_type,
)


def test_list_runs_by_type_filters_correctly(tmp_path: Path) -> None:
    reg = tmp_path / "run_registry.jsonl"
    append_run_metadata(
        ExperimentMetadata(
            run_id="baseline-1",
            run_type="baseline",
            code_revision="abc",
            config_ref="configs/baseline.yaml",
            seed=42,
            fold_scheme="spatial_blocked",
            command="train",
            artifacts={"model": "m1"},
        ),
        reg,
    )
    append_run_metadata(
        ExperimentMetadata(
            run_id="candidate-1",
            run_type="candidate",
            code_revision="abc",
            config_ref="configs/candidate.yaml",
            seed=42,
            fold_scheme="spatial_blocked",
            command="train",
            artifacts={"model": "m2"},
        ),
        reg,
    )

    baselines = list_runs_by_type("baseline", reg)

    assert len(baselines) == 1
    assert baselines[0]["run_id"] == "baseline-1"


def test_get_run_lineage_returns_related_runs(tmp_path: Path) -> None:
    reg = tmp_path / "run_registry.jsonl"
    first = ExperimentMetadata(
        run_id="baseline-1",
        run_type="baseline",
        code_revision="abc",
        config_ref="configs/baseline.yaml",
        seed=42,
        fold_scheme="spatial_blocked",
        command="train",
        artifacts={"model": "m1"},
    )
    second = ExperimentMetadata(
        run_id="baseline-2",
        run_type="baseline",
        code_revision="abc",
        config_ref="configs/baseline.yaml",
        seed=42,
        fold_scheme="spatial_blocked",
        command="train",
        artifacts={"model": "m2"},
    )
    append_run_metadata(first, reg)
    append_run_metadata(second, reg)

    lineage = get_run_lineage("baseline-2", reg)

    assert len(lineage) >= 1
    assert any(item["run_id"] == "baseline-1" for item in lineage)
    assert find_run("baseline-2", reg) is not None
