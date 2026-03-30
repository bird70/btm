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


# ---------------------------------------------------------------------------
# T006: FeatureFlags and PipelineConfig extension tests (RED phase)
# ---------------------------------------------------------------------------


def test_pipeline_config_accepts_model_type() -> None:
    """model_type is parsed from YAML dict and stored on PipelineConfig."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict({"seed": 42, "model_type": "lgbm"})
    assert cfg.model_type == "lgbm"


def test_pipeline_config_model_type_defaults_to_none() -> None:
    """When model_type is absent the field defaults to None."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict({"seed": 42})
    assert cfg.model_type is None


def test_pipeline_config_invalid_model_type_raises() -> None:
    """Unknown model_type values raise ValueError."""
    import pytest

    from benthic_model.config import PipelineConfig

    with pytest.raises(ValueError, match="model_type"):
        PipelineConfig.from_dict({"seed": 42, "model_type": "nonsense_model"})


def test_pipeline_config_accepts_feature_flags() -> None:
    """feature_flags mapping is parsed into a FeatureFlags instance."""
    from benthic_model.config import FeatureFlags, PipelineConfig

    cfg = PipelineConfig.from_dict(
        {
            "seed": 42,
            "feature_flags": {
                "include_focal_stats": False,
                "include_interactions": True,
                "include_spatial_z_scores": False,
                "include_btm_features": True,
            },
        }
    )
    assert isinstance(cfg.feature_flags, FeatureFlags)
    assert cfg.feature_flags.include_focal_stats is False
    assert cfg.feature_flags.include_interactions is True
    assert cfg.feature_flags.include_spatial_z_scores is False
    assert cfg.feature_flags.include_btm_features is True


def test_pipeline_config_feature_flags_defaults_to_none() -> None:
    """When feature_flags key is absent, field is None (backwards-compatible)."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict({"seed": 42})
    assert cfg.feature_flags is None


def test_feature_flags_all_default_true() -> None:
    """FeatureFlags constructed with no arguments has all fields True."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags()
    assert flags.include_focal_stats is True
    assert flags.include_interactions is True
    assert flags.include_spatial_z_scores is True
    assert flags.include_btm_features is True


def test_experiment_metadata_has_model_type_used_field() -> None:
    """ExperimentMetadata accepts model_type_used and feature_flags_used fields."""
    metadata = ExperimentMetadata(
        run_id="run-002",
        run_type="baseline",
        code_revision="abc123",
        config_ref="configs/baseline.yaml",
        seed=42,
        fold_scheme="spatial_blocked",
        command="train",
        artifacts={},
        model_type_used="rf",
        feature_flags_used={"include_focal_stats": True, "include_interactions": False},
    )
    assert metadata.model_type_used == "rf"
    assert metadata.feature_flags_used["include_interactions"] is False


def test_experiment_metadata_model_type_used_defaults_empty() -> None:
    """model_type_used defaults to empty string (backwards-compatible)."""
    metadata = ExperimentMetadata(
        run_id="run-003",
        run_type="baseline",
        code_revision="abc123",
        config_ref="configs/baseline.yaml",
        seed=42,
        fold_scheme="spatial_blocked",
        command="train",
        artifacts={},
    )
    assert metadata.model_type_used == ""
    assert metadata.feature_flags_used == {}


# ---------------------------------------------------------------------------
# T004: include_eco_features + model_params — RED phase tests
# These tests must FAIL before T006 implementation changes are made.
# ---------------------------------------------------------------------------


def test_feature_flags_has_include_eco_features_field() -> None:
    """FeatureFlags must have include_eco_features defaulting to False (T002)."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags()
    assert hasattr(flags, "include_eco_features"), (
        "FeatureFlags missing 'include_eco_features' field"
    )
    assert flags.include_eco_features is False


def test_feature_flags_include_eco_features_can_be_set_true() -> None:
    """FeatureFlags.include_eco_features can be set to True."""
    from benthic_model.config import FeatureFlags

    flags = FeatureFlags(include_eco_features=True)
    assert flags.include_eco_features is True


def test_pipeline_config_from_dict_parses_include_eco_features() -> None:
    """PipelineConfig.from_dict passes include_eco_features through feature_flags."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict(
        {
            "seed": 42,
            "feature_flags": {
                "include_btm_features": True,
                "include_eco_features": True,
            },
        }
    )
    assert cfg.feature_flags is not None
    assert cfg.feature_flags.include_eco_features is True


def test_pipeline_config_has_model_params_field() -> None:
    """PipelineConfig must have model_params field defaulting to None (T003)."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict({"seed": 42})
    assert hasattr(cfg, "model_params"), "PipelineConfig missing 'model_params' field"
    assert cfg.model_params is None


def test_pipeline_config_from_dict_passes_model_params_through() -> None:
    """PipelineConfig.from_dict parses model_params as a plain dict."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict(
        {
            "seed": 42,
            "model_type": "rf",
            "model_params": {"n_estimators": 500, "max_features": "sqrt"},
        }
    )
    assert cfg.model_params is not None
    assert cfg.model_params["n_estimators"] == 500
    assert cfg.model_params["max_features"] == "sqrt"


def test_pipeline_config_model_params_defaults_none_when_absent() -> None:
    """model_params defaults to None when not in config dict (backwards-compatible)."""
    from benthic_model.config import PipelineConfig

    cfg = PipelineConfig.from_dict({"seed": 42, "model_type": "rf"})
    assert cfg.model_params is None
