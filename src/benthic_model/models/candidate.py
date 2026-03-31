from __future__ import annotations

import os

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None


def _cuda_available() -> bool:
    """Return True when GPU acceleration should be used.

    Checks XGBoost's compile-time ``USE_CUDA`` flag as a fast proxy for
    CUDA availability (avoids a heavyweight driver probe on every import).

    Override via environment variable:
    - ``BTM_USE_GPU=0``  — force CPU mode (useful on CPU-only machines)
    - ``BTM_USE_GPU=1``  — force GPU mode
    - unset / ``auto``   — auto-detect (default)
    """
    env = os.environ.get("BTM_USE_GPU", "auto").lower()
    if env == "0":
        return False
    if env == "1":
        return True
    try:
        from xgboost import build_info  # noqa: PLC0415

        return bool(build_info().get("USE_CUDA"))
    except Exception:
        return False


class CandidateXGBoostModel:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._encoder = LabelEncoder()
        self._xgb = None
        self._gbdt = GradientBoostingClassifier(random_state=seed)

    def fit(self, X, y):
        encoded = self._encoder.fit_transform(y)
        if XGBClassifier is not None:
            device = "cuda" if _cuda_available() else "cpu"
            self._xgb = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softmax",
                num_class=len(self._encoder.classes_),
                random_state=self._seed,
                n_jobs=-1,  # was 1 — use all cores in CPU mode
                tree_method="hist",
                device=device,
            )
            self._xgb.fit(X, encoded)
        else:
            self._gbdt.fit(X, encoded)
        return self

    def predict(self, X):
        if self._xgb is not None:
            encoded = self._xgb.predict(X)
        else:
            encoded = self._gbdt.predict(X)
        return self._encoder.inverse_transform(encoded.astype(int))


def build_candidate_model(seed: int = 42) -> CandidateXGBoostModel:
    return CandidateXGBoostModel(seed=seed)


class CandidateLGBMModel:
    """LightGBM multi-class classifier wrapper."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        if LGBMClassifier is None:  # pragma: no cover
            raise ImportError("lightgbm is required for CandidateLGBMModel")
        device = "gpu" if _cuda_available() else "cpu"
        self._model = LGBMClassifier(
            n_estimators=600,
            learning_rate=0.03,
            num_leaves=63,
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            device=device,
            n_jobs=-1,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)


def build_lgbm_model(seed: int = 42) -> CandidateLGBMModel:
    return CandidateLGBMModel(seed=seed)


class CandidateCatBoostModel:
    """CatBoost multi-class classifier wrapper."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        if CatBoostClassifier is None:  # pragma: no cover
            raise ImportError("catboost is required for CandidateCatBoostModel")
        task_type = "GPU" if _cuda_available() else "CPU"
        self._model = CatBoostClassifier(
            iterations=800,
            depth=7,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=seed,
            task_type=task_type,
        )

    def fit(self, X, y):
        self._model.fit(X, y)
        return self

    def predict(self, X):
        result = np.array(self._model.predict(X)).ravel().astype(str)
        return result


def build_catboost_model(seed: int = 42) -> CandidateCatBoostModel:
    return CandidateCatBoostModel(seed=seed)


class CandidateEnsembleModel:
    """Soft-vote ensemble of a Random Forest and a LightGBM classifier."""

    def __init__(self, seed: int = 42) -> None:
        from benthic_model.models.baseline import build_baseline_model

        self._rf = build_baseline_model(seed=seed)
        self._lgbm = CandidateLGBMModel(seed=seed)
        self._classes: list[str] = []

    def fit(self, X, y):
        self._classes = (
            sorted(y.unique().tolist()) if hasattr(y, "unique") else sorted(set(y))
        )
        self._rf.fit(X, y)
        self._lgbm.fit(X, y)
        return self

    def predict(self, X):
        rf_proba = self._rf.model.predict_proba(X)
        lgbm_proba = self._lgbm._model.predict_proba(X)
        mean_proba = (np.array(rf_proba) + np.array(lgbm_proba)) / 2.0
        indices = np.argmax(mean_proba, axis=1)
        classes = self._rf.model.classes_
        return np.array(classes)[indices]


def build_rf_lgbm_ensemble_model(seed: int = 42) -> CandidateEnsembleModel:
    return CandidateEnsembleModel(seed=seed)
