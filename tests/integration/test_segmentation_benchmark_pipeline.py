from pathlib import Path

from benthic_model.cli import main
from tests.helpers import create_train_test_csvs


def _write_seg_config(path: Path, train_csv: Path, test_csv: Path, out_dir: Path) -> Path:
    cfg = path / "segmentation.yaml"
    cfg.write_text(
        "input:\n"
        f"  train_csv: {train_csv}\n"
        f"  test_csv: {test_csv}\n"
        "output:\n"
        f"  base_dir: {out_dir}\n"
        "split:\n"
        "  val_fraction: 0.2\n"
        "  random_state: 42\n"
        "weights:\n"
        "  strategy: inverse_freq_capped\n"
        "  cap: 5.0\n"
        "candidates:\n"
        "  - segformer_ft\n"
        "  - segformer_ft_crf\n"
        "  - deeplabv3_ft\n",
        encoding="utf-8",
    )
    return cfg


def test_segmentation_benchmark_pipeline_integration(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv, test_csv, _ = create_train_test_csvs(data_dir)

    seg_base = tmp_path / "artifacts" / "segmentation"
    cfg = _write_seg_config(tmp_path, train_csv, test_csv, seg_base)

    assert main(["segmentation-benchmark", "--config", str(cfg)]) == 0
    run_dir = sorted(seg_base.glob("seg-*"))[-1]

    assert (run_dir / "candidate_metrics.json").exists()
    assert (run_dir / "val_location_predictions.csv").exists()
    assert (run_dir / "test_location_predictions.csv").exists()
    assert (run_dir / "val_truth.csv").exists()
