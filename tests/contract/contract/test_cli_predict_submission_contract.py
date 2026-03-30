from pathlib import Path

import pandas as pd

from benthic_model.cli import main
from tests.helpers import (
    create_basic_config,
    create_mbes_rasters,
    create_metadata_file,
    create_train_test_csvs,
    latest_run_id,
)


def test_cli_predict_and_make_submission_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_csv, test_csv, sample_submission = create_train_test_csvs(data_dir)
    bathy_tif, backscatter_tif = create_mbes_rasters(data_dir)
    create_metadata_file(data_dir)

    baseline_cfg = create_basic_config(tmp_path / "configs", run_type="baseline")

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

    run_id = latest_run_id(tmp_path / "artifacts/experiments/run_registry.jsonl")

    assert (
        main(
            [
                "predict",
                "--run-id",
                run_id,
                "--test-csv",
                str(test_csv),
                "--bathymetry-tif",
                str(bathy_tif),
                "--backscatter-tif",
                str(backscatter_tif),
            ]
        )
        == 0
    )

    predictions = tmp_path / "artifacts/predictions" / f"{run_id}_test_predictions.csv"
    submission = tmp_path / "submissions" / "submission.csv"

    assert (
        main(
            [
                "make-submission",
                "--predictions",
                str(predictions),
                "--sample-submission",
                str(sample_submission),
                "--output",
                str(submission),
            ]
        )
        == 0
    )

    frame = pd.read_csv(submission)
    assert list(frame.columns) == ["ID", "class"]
