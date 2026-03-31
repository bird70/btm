"""Unit tests for src.benthic_model.features.selection — permutation importance selection."""

from __future__ import annotations

import logging
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# Import path for the module under test
from benthic_model.features.selection import (
    FeatureSelectionResult,
    select_by_permutation_importance,
)


def _make_synthetic(
    n_samples: int = 50,
    n_informative: int = 5,
    n_noise: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) with n_informative useful features and n_noise random ones."""
    rng = np.random.default_rng(seed)
    n_classes = 3
    n_features = n_informative + n_noise
    X_arr = rng.standard_normal((n_samples, n_features))
    feature_names = [f"feat_{i}" for i in range(n_features)]
    X = pd.DataFrame(X_arr, columns=feature_names)
    # Labels determined only by the informative features
    y = pd.Series(
        (X_arr[:, :n_informative].sum(axis=1) > 0).astype(int) % n_classes,
        name="label",
    )
    return X, y


def _fit_model(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=20, random_state=0, max_depth=4)
    clf.fit(X, y)
    return clf


class TestSelectByPermutationImportance:
    def test_returns_feature_selection_result(self):
        X, y = _make_synthetic(50, 5, 5)
        model = _fit_model(X, y)
        result = select_by_permutation_importance(X, y, model, n_repeats=3)
        assert isinstance(result, FeatureSelectionResult)
        assert isinstance(result.selected_features, list)
        assert isinstance(result.dropped_features, list)
        assert isinstance(result.importance_df, pd.DataFrame)

    def test_importance_df_schema(self):
        X, y = _make_synthetic(50, 5, 5)
        model = _fit_model(X, y)
        result = select_by_permutation_importance(X, y, model, n_repeats=3)
        expected_cols = {
            "feature_name",
            "importance_mean",
            "importance_std",
            "selected",
            "rank",
        }
        assert expected_cols.issubset(set(result.importance_df.columns))
        assert len(result.importance_df) == len(X.columns)

    def test_selected_plus_dropped_equals_all_features(self):
        X, y = _make_synthetic(50, 5, 5)
        model = _fit_model(X, y)
        result = select_by_permutation_importance(X, y, model, n_repeats=3)
        all_features = set(X.columns)
        retained = set(result.selected_features)
        dropped = set(result.dropped_features)
        assert retained | dropped == all_features

    def test_reduces_features_with_noise(self):
        """With 10 pure noise features, selection should drop several."""
        X, y = _make_synthetic(n_samples=80, n_informative=5, n_noise=10, seed=7)
        model = _fit_model(X, y)
        result = select_by_permutation_importance(
            X, y, model, n_repeats=5, importance_threshold=0.0
        )
        # Should retain fewer than all 15 features
        assert len(result.selected_features) < len(X.columns)

    def test_logs_dropped_features(self):
        """Logger should emit at least one log.info call that names a dropped feature."""
        X, y = _make_synthetic(50, 5, 5, seed=42)
        model = _fit_model(X, y)

        with patch("benthic_model.features.selection._log") as mock_log:
            result = select_by_permutation_importance(X, y, model, n_repeats=3)

        # Collect all log.info calls
        all_logged = " ".join(
            str(c) for call in mock_log.info.call_args_list for c in call[0]
        )
        # At least one dropped feature name should appear in the logs
        if result.dropped_features:
            assert any(f in all_logged for f in result.dropped_features)

    def test_correlation_prefilter_drops_redundant(self):
        """Two features with |r| > 0.95 — lower-variance one should be dropped."""
        rng = np.random.default_rng(0)
        base = rng.standard_normal(60)
        # feat_a: high variance
        feat_a = base * 10.0
        # feat_b: almost identical but lower variance
        feat_b = base * 1.0 + rng.standard_normal(60) * 0.01

        X = pd.DataFrame(
            {"feat_a": feat_a, "feat_b": feat_b, "feat_c": rng.standard_normal(60)}
        )
        y = pd.Series((feat_a > 0).astype(int), name="label")
        model = _fit_model(X, y)

        result = select_by_permutation_importance(
            X, y, model, n_repeats=3, corr_threshold=0.95
        )
        # feat_b (lower variance) must be dropped; feat_a must be retained
        assert "feat_b" in result.dropped_features
        assert "feat_a" not in result.dropped_features
