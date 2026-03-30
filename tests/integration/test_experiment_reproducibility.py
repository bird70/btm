from pathlib import Path

from benthic_model.cli import main
from benthic_model.experiment.compare_runs import compare_runs_with_tolerance
from tests.helpers import (
    create_basic_config,
    create_mbes_rasters,
    create_metadata_file,
    create_train_test_csvs,
    latest_run_id,
)


def test_experiment_rerun_reproducibility(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_csv, _, _ = create_train_test_csvs(data_dir)
    bathy_tif, backscatter_tif = create_mbes_rasters(data_dir)
    create_metadata_file(data_dir)
    cfg = create_basic_config(tmp_path / "configs", run_type="baseline")

    base_args = [
        "train",
        "--train-csv",
        str(train_csv),
        "--bathymetry-tif",
        str(bathy_tif),
        "--backscatter-tif",
        str(backscatter_tif),
        "--config",
        str(cfg),
        "--run-type",
        "baseline",
        "--seed",
        "42",
    ]

    assert main(base_args) == 0
    run1 = latest_run_id(tmp_path / "artifacts" / "experiments" / "run_registry.jsonl")

    assert main(base_args) == 0
    run2 = latest_run_id(tmp_path / "artifacts" / "experiments" / "run_registry.jsonl")

    summary = compare_runs_with_tolerance(run1, run2, tolerance=0.01)
    assert summary["within_tolerance"] is True
