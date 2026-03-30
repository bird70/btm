from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class CrossValidationConfig:
    fold_scheme: str = "spatial_blocked"
    n_splits: int = 5
    random_state: int = 42
    spatial_bins: int = 4


@dataclass(slots=True)
class PipelineConfig:
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    reports_dir: str = "reports"
    submissions_dir: str = "submissions"
    seed: int = 42
    cv: CrossValidationConfig = field(default_factory=CrossValidationConfig)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> PipelineConfig:
        cv_cfg = config.get("cv", {})
        cv = CrossValidationConfig(**cv_cfg)
        payload = {k: v for k, v in config.items() if k != "cv"}
        return cls(cv=cv, **payload)

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
