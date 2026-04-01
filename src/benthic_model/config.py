from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

_ALLOWED_MODEL_TYPES = frozenset({"rf", "xgb", "lgbm", "catboost", "rf_lgbm_ensemble"})


@dataclass(slots=True)
class CrossValidationConfig:
    fold_scheme: str = "spatial_blocked"
    n_splits: int = 5
    random_state: int = 42
    spatial_bins: int = 4


@dataclass(slots=True)
class FeatureFlags:
    """Controls which derived feature groups are added by engineer_features()."""

    include_focal_stats: bool = True
    include_interactions: bool = True
    include_spatial_z_scores: bool = True
    include_btm_features: bool = True
    include_eco_features: bool = False
    exclude_coords: bool = False


@dataclass(slots=True)
class PipelineConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    submissions_dir: str = "submissions"
    seed: int = 42
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    model_type: str | None = None
    feature_flags: FeatureFlags | None = None
    model_params: dict[str, Any] | None = None
    # informational; not consumed by the pipeline
    spatial_coords: bool = False

    def __post_init__(self) -> None:
        if self.model_type is not None and self.model_type not in _ALLOWED_MODEL_TYPES:
            raise ValueError(
                f"Invalid model_type {self.model_type!r}. "
                f"Allowed values: {sorted(_ALLOWED_MODEL_TYPES)}"
            )

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PipelineConfig:
        cv_cfg = config.get("cv", {})
        cv = CrossValidationConfig(**cv_cfg)

        flags_raw = config.get("feature_flags")
        feature_flags: FeatureFlags | None = None
        if flags_raw is not None:
            feature_flags = FeatureFlags(**flags_raw)

        model_params: dict[str, Any] | None = config.get("model_params")

        ignored = {"cv", "feature_flags", "model_params"}
        payload = {k: v for k, v in config.items() if k not in ignored}
        return cls(
            cv=cv, feature_flags=feature_flags, model_params=model_params, **payload
        )

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> PipelineConfig:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Configuration file must contain a mapping at root.")
        return cls.from_dict(loaded)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
