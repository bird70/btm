from pathlib import Path

import pytest

from benthic_model.segmentation.benchmark import CANDIDATES, run_segmentation_benchmark
from tests.helpers import create_train_test_csvs


def test_candidate_registry_is_exact() -> None:
    assert CANDIDATES == ("segformer_ft", "segformer_ft_crf", "deeplabv3_ft")


def test_benchmark_rejects_non_exact_candidates(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    train_csv, test_csv, _ = create_train_test_csvs(data_dir)

    cfg = tmp_path / "segmentation.yaml"
    cfg.write_text(
        "input:\n"
        f"  train_csv: {train_csv}\n"
        f"  test_csv: {test_csv}\n"
        "output:\n"
        f"  base_dir: {tmp_path / 'artifacts' / 'segmentation'}\n"
        "split:\n"
        "  val_fraction: 0.2\n"
        "  random_state: 42\n"
        "weights:\n"
        "  strategy: inverse_freq_capped\n"
        "  cap: 5.0\n"
        "candidates:\n"
        "  - segformer_ft\n"
        "  - deeplabv3_ft\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        run_segmentation_benchmark(cfg)
