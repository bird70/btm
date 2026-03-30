"""T008: Failing tests for new model builders (Red phase).

Each builder is tested on a minimal 20-row synthetic 5-class dataset.
predict() must return labels exclusively within the benthic class vocabulary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

CLASSES = ["ALG", "FMAT", "NVB", "SGAM", "SGZ"]


def _synthetic_data(n: int = 20, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, 4)),
        columns=["bathymetry", "backscatter", "depth_std", "back_std"],
    )
    y = pd.Series([CLASSES[i % len(CLASSES)] for i in range(n)])
    return X, y


def test_build_lgbm_model_trains_and_predicts() -> None:
    """build_lgbm_model() returns a model that fits and predicts class labels."""
    from benthic_model.models.candidate import build_lgbm_model

    X, y = _synthetic_data()
    model = build_lgbm_model(seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset(
        CLASSES
    ), f"Unexpected labels: {set(preds) - set(CLASSES)}"


def test_build_catboost_model_trains_and_predicts() -> None:
    """build_catboost_model() returns a model that fits and predicts class labels."""
    from benthic_model.models.candidate import build_catboost_model

    X, y = _synthetic_data()
    model = build_catboost_model(seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset(
        CLASSES
    ), f"Unexpected labels: {set(preds) - set(CLASSES)}"


def test_build_rf_lgbm_ensemble_model_trains_and_predicts() -> None:
    """build_rf_lgbm_ensemble_model() returns a soft-vote ensemble that predicts class labels."""
    from benthic_model.models.candidate import build_rf_lgbm_ensemble_model

    X, y = _synthetic_data()
    model = build_rf_lgbm_ensemble_model(seed=42)
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)
    assert set(preds).issubset(
        CLASSES
    ), f"Unexpected labels: {set(preds) - set(CLASSES)}"


def test_lgbm_model_seed_determinism() -> None:
    """Two models with the same seed produce identical predictions."""
    from benthic_model.models.candidate import build_lgbm_model

    X, y = _synthetic_data()
    m1, m2 = build_lgbm_model(seed=7), build_lgbm_model(seed=7)
    m1.fit(X, y)
    m2.fit(X, y)
    assert list(m1.predict(X)) == list(m2.predict(X))
