from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class ExperimentMetadata:
    run_id: str
    run_type: str
    code_revision: str
    config_ref: str
    seed: int
    fold_scheme: str
    command: str
    artifacts: dict[str, str]
    config_hash: str = ""
    data_split_ref: str = ""
    metric_weighted_f1: float | None = None
    metric_per_class_f1: dict[str, float] = field(default_factory=dict)
    model_type_used: str = ""
    feature_flags_used: dict[str, bool] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentMetadata:
        return cls(**payload)
