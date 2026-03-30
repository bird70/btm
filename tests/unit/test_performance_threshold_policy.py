import pytest

from benthic_model.evaluation.compare import enforce_weighted_f1_threshold


def test_threshold_policy_raises_for_significant_degradation() -> None:
    with pytest.raises(ValueError, match="degraded"):
        enforce_weighted_f1_threshold(
            baseline_weighted_f1=0.82,
            candidate_weighted_f1=0.80,
            threshold=0.005,
            override=False,
        )


def test_threshold_policy_allows_override() -> None:
    enforce_weighted_f1_threshold(
        baseline_weighted_f1=0.82,
        candidate_weighted_f1=0.80,
        threshold=0.005,
        override=True,
    )
