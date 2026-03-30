from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None


class CandidateXGBoostModel:
    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._encoder = LabelEncoder()
        self._xgb = None
        self._gbdt = GradientBoostingClassifier(random_state=seed)

    def fit(self, X, y):
        encoded = self._encoder.fit_transform(y)
        if XGBClassifier is not None:
            self._xgb = XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softmax",
                num_class=len(self._encoder.classes_),
                random_state=self._seed,
                n_jobs=1,
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
