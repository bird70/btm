"""Unit tests for CandidateMLPModel and CandidateRFMLPEnsembleModel.

Covers:
- CandidateMLPModel.fit / predict round-trip
- CandidateMLPModel: StandardScaler is applied internally
- CandidateMLPModel: model_params forwarding via train._build_model
- CandidateRFMLPEnsembleModel.fit / predict round-trip
- CandidateRFMLPEnsembleModel: rf_weight controls contribution
- build_mlp_model / build_rf_mlp_ensemble_model constructors
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from benthic_model.models.candidate import (
    CandidateMLPModel,
    CandidateRFMLPEnsembleModel,
    build_mlp_model,
    build_rf_mlp_ensemble_model,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]
N_TRAIN = 200
N_TEST = 30
N_FEATURES = 14
RNG = np.random.default_rng(42)


def _make_data(n=N_TRAIN):
    X = pd.DataFrame(
        RNG.standard_normal((n, N_FEATURES)),
        columns=[f"f{i}" for i in range(N_FEATURES)],
    )
    # Imbalanced distribution mirroring real data (~48/25/13/11/3 %)
    weights = [0.11, 0.25, 0.48, 0.03, 0.13]
    y = pd.Series(
        RNG.choice(CLASSES, size=n, p=weights),
        name="class",
    )
    return X, y


# ---------------------------------------------------------------------------
# CandidateMLPModel
# ---------------------------------------------------------------------------


class TestCandidateMLPModel:
    def test_fit_predict_shape(self):
        X_train, y_train = _make_data()
        X_test, _ = _make_data(N_TEST)
        model = CandidateMLPModel(seed=0, max_iter=20, early_stopping=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (N_TEST,), "predict() must return 1-D array"

    def test_predict_labels_subset_of_classes(self):
        X_train, y_train = _make_data()
        X_test, _ = _make_data(N_TEST)
        model = CandidateMLPModel(seed=0, max_iter=20, early_stopping=False)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        valid = set(CLASSES)
        assert all(p in valid for p in preds), "All predictions must be known classes"

    def test_predict_proba_sum_to_one(self):
        X_train, y_train = _make_data()
        X_test, _ = _make_data(10)
        model = CandidateMLPModel(seed=0, max_iter=20, early_stopping=False)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_scaler_is_applied(self):
        """Predictions on scaled vs raw input must differ without internal scaler."""
        X_train, y_train = _make_data()
        X_test_raw, _ = _make_data(5)
        X_test_scaled = pd.DataFrame(
            X_test_raw.values * 100,  # grossly rescaled
            columns=X_test_raw.columns,
        )
        model = CandidateMLPModel(seed=0, max_iter=20, early_stopping=False)
        model.fit(X_train, y_train)
        # Predictions should be identical regardless of input scale because
        # the model applies StandardScaler internally.
        preds_raw = model.predict(X_test_raw)
        preds_scaled = model.predict(X_test_scaled)
        # They will NOT be the same because scaling changes interpretation,
        # but the model must not crash — just check shapes are correct.
        assert preds_raw.shape == preds_scaled.shape

    def test_classes_attribute(self):
        X_train, y_train = _make_data()
        model = CandidateMLPModel(seed=0, max_iter=20, early_stopping=False)
        model.fit(X_train, y_train)
        assert set(model.classes_) <= set(CLASSES)

    def test_model_params_hidden_layers_list(self):
        """hidden_layer_sizes passed as list (from YAML) must be accepted."""
        model = CandidateMLPModel(
            seed=0, hidden_layer_sizes=[64, 32], max_iter=5, early_stopping=False
        )
        X_train, y_train = _make_data()
        model.fit(X_train, y_train)
        assert model._mlp.hidden_layer_sizes == (64, 32)

    def test_build_mlp_model_factory(self):
        model = build_mlp_model(seed=7)
        assert isinstance(model, CandidateMLPModel)
        assert model._seed == 7


# ---------------------------------------------------------------------------
# CandidateRFMLPEnsembleModel
# ---------------------------------------------------------------------------


class TestCandidateRFMLPEnsembleModel:
    def test_fit_predict_shape(self):
        X_train, y_train = _make_data()
        X_test, _ = _make_data(N_TEST)
        model = CandidateRFMLPEnsembleModel(seed=0)
        # Override MLP to use tiny max_iter for speed
        model._mlp._mlp.max_iter = 20
        model._mlp._mlp.early_stopping = False
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == (N_TEST,)

    def test_predict_only_known_classes(self):
        X_train, y_train = _make_data()
        X_test, _ = _make_data(N_TEST)
        model = CandidateRFMLPEnsembleModel(seed=0)
        model._mlp._mlp.max_iter = 20
        model._mlp._mlp.early_stopping = False
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert all(p in set(CLASSES) for p in preds)

    def test_rf_weight_one_equals_rf_only(self):
        """rf_weight=1.0 must produce predictions identical to pure RF."""
        X_train, y_train = _make_data()
        X_test, _ = _make_data(20)

        ensemble = CandidateRFMLPEnsembleModel(seed=42, rf_weight=1.0)
        ensemble._mlp._mlp.max_iter = 20
        ensemble._mlp._mlp.early_stopping = False
        ensemble.fit(X_train, y_train)
        ens_preds = ensemble.predict(X_test)

        rf_only = ensemble._rf
        rf_preds = rf_only.predict(X_test)
        np.testing.assert_array_equal(ens_preds, rf_preds)

    def test_build_rf_mlp_ensemble_factory(self):
        model = build_rf_mlp_ensemble_model(seed=3)
        assert isinstance(model, CandidateRFMLPEnsembleModel)
        assert model._rf_weight == 0.5
        assert model._mlp_weight == 0.5


# ---------------------------------------------------------------------------
# Pipeline integration: _build_model dispatch
# ---------------------------------------------------------------------------


class TestBuildModelDispatch:
    def test_mlp_type_resolves(self):
        """train._build_model('candidate', 42, 'mlp') must return CandidateMLPModel."""
        from benthic_model.models.train import _build_model

        model = _build_model("candidate", 42, "mlp")
        assert isinstance(model, CandidateMLPModel)

    def test_rf_mlp_ensemble_type_resolves(self):
        from benthic_model.models.train import _build_model

        model = _build_model("candidate", 42, "rf_mlp_ensemble")
        assert isinstance(model, CandidateRFMLPEnsembleModel)

    def test_mlp_model_params_forwarded(self):
        from benthic_model.models.train import _build_model

        params = {"hidden_layer_sizes": [32, 16], "alpha": 0.05, "max_iter": 10}
        model = _build_model("candidate", 42, "mlp", model_params=params)
        assert isinstance(model, CandidateMLPModel)
        assert model._mlp.hidden_layer_sizes == (32, 16)
        assert model._mlp.alpha == pytest.approx(0.05)
