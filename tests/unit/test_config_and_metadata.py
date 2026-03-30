from benthic_model.config import PipelineConfig
from benthic_model.experiment.metadata import ExperimentMetadata


def test_pipeline_config_from_dict_reads_nested_cv() -> None:
    cfg = PipelineConfig.from_dict(
        {
            "seed": 123,
            "data_dir": "dataset",
            "cv": {"n_splits": 4, "fold_scheme": "spatial_blocked", "spatial_bins": 3},
        }
    )

    assert cfg.seed == 123
    assert cfg.data_dir == "dataset"
    assert cfg.cv.n_splits == 4
    assert cfg.cv.spatial_bins == 3


def test_experiment_metadata_round_trip() -> None:
    metadata = ExperimentMetadata(
        run_id="run-001",
        run_type="baseline",
        code_revision="abc123",
        config_ref="configs/baseline.yaml",
        seed=42,
        fold_scheme="spatial_blocked",
        command="train --run-type baseline",
        artifacts={"model": "artifacts/models/model.joblib"},
        metric_weighted_f1=0.81,
        metric_per_class_f1={"ALG": 0.79},
    )

    clone = ExperimentMetadata.from_dict(metadata.to_dict())

    assert clone.run_id == "run-001"
    assert clone.metric_weighted_f1 == 0.81
    assert clone.metric_per_class_f1["ALG"] == 0.79
