from pathlib import Path

from benthic_model.cli import main
from tests.helpers import (
    create_basic_config,
    create_mbes_rasters,
    create_metadata_file,
    create_train_test_csvs,
    latest_run_id,
)


def test_training_evaluation_pipeline_integration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_csv, _, _ = create_train_test_csvs(data_dir)
    bathy_tif, backscatter_tif = create_mbes_rasters(data_dir)
    create_metadata_file(data_dir)

    baseline_cfg = create_basic_config(tmp_path / "configs", run_type="baseline")
    candidate_cfg = create_basic_config(tmp_path / "configs", run_type="candidate")

    assert (
        main(
            [
                "train",
                "--train-csv",
                str(train_csv),
                "--bathymetry-tif",
                str(bathy_tif),
                "--backscatter-tif",
                str(backscatter_tif),
                "--config",
                str(baseline_cfg),
                "--run-type",
                "baseline",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "train",
                "--train-csv",
                str(train_csv),
                "--bathymetry-tif",
                str(bathy_tif),
                "--backscatter-tif",
                str(backscatter_tif),
                "--config",
                str(candidate_cfg),
                "--run-type",
                "candidate",
            ]
        )
        == 0
    )

    run_id = latest_run_id(tmp_path / "artifacts/experiments/run_registry.jsonl")
    assert main(["evaluate", "--run-id", run_id, "--fold-scheme", "spatial_blocked"]) == 0

    report = tmp_path / "reports" / "metrics" / f"{run_id}_comparison.md"
    assert report.exists()
