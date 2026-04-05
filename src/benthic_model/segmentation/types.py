from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CandidateRunMetrics:
    candidate_type: str
    status: str
    weighted_f1: float | None
    macro_f1: float | None
    sgam_recall: float | None
    runtime_minutes: float
    class_weight_strategy: str
    class_weight_cap: float
    message: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PromotionDecision:
    baseline_weighted_f1: float
    hybrid_weighted_f1: float
    baseline_sgam_recall: float
    hybrid_sgam_recall: float
    gate_passed: bool
    decision_reason: str

    @property
    def weighted_f1_delta(self) -> float:
        return self.hybrid_weighted_f1 - self.baseline_weighted_f1

    @property
    def sgam_recall_delta(self) -> float:
        return self.hybrid_sgam_recall - self.baseline_sgam_recall

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["weighted_f1_delta"] = self.weighted_f1_delta
        payload["sgam_recall_delta"] = self.sgam_recall_delta
        return payload
