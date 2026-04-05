import pytest

from benthic_model.segmentation.types import PromotionDecision


def test_promotion_decision_deltas() -> None:
    decision = PromotionDecision(
        baseline_weighted_f1=0.70,
        hybrid_weighted_f1=0.73,
        baseline_sgam_recall=0.40,
        hybrid_sgam_recall=0.45,
        gate_passed=True,
        decision_reason="Hybrid promoted",
    )

    payload = decision.to_dict()
    assert payload["weighted_f1_delta"] == pytest.approx(0.03)
    assert payload["sgam_recall_delta"] == pytest.approx(0.05)
    assert payload["gate_passed"] is True
