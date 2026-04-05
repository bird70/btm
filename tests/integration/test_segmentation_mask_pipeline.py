from pathlib import Path

from benthic_model.cli import main
from tests.helpers import create_train_test_csvs


def test_segmentation_mask_pipeline_integration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv, _, _ = create_train_test_csvs(data_dir)

    out_dir = tmp_path / "artifacts" / "segmentation" / "mask-run"
    assert (
        main(
            [
                "segmentation-build-masks",
                "--train-csv",
                str(train_csv),
                "--output-dir",
                str(out_dir),
            ]
        )
        == 0
    )

    assert (out_dir / "masks.npy").exists()
    assert (out_dir / "mask_metadata.json").exists()
