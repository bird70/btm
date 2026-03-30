from pathlib import Path

from benthic_model.cli import main
from tests.helpers import (
    create_basic_config,
    create_mbes_rasters,
    create_metadata_file,
    create_train_test_csvs,
)


def test_experiment_metadata_contains_required_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    train_csv, _, _ = create_train_test_csvs(data_dir)
    bathy_tif, backscatter_tif = create_mbes_rasters(data_dir)
    create_metadata_file(data_dir)
    cfg = create_basic_config(tmp_path / "configs", run_type="baseline")

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
                str(cfg),
                "--run-type",
                "baseline",
            ]
        )
        == 0
    )

    registry_path = tmp_path / "artifacts" / "experiments" / "run_registry.jsonl"
    line = registry_path.read_text(encoding="utf-8").splitlines()[-1]

    required_fields = [
        "run_id",
        "run_type",
        "timestamp",
        "code_revision",
        "config_ref",
        "seed",
        "fold_scheme",
        "command",
        "artifacts",
        "config_hash",
        "data_split_ref",
    ]

    for field in required_fields:
        assert f'"{field}"' in line
