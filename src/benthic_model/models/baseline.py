from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier


class BaselineRandomForestModel:
    def __init__(self, seed: int = 42) -> None:
        self.model = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def build_baseline_model(seed: int = 42) -> BaselineRandomForestModel:
    return BaselineRandomForestModel(seed=seed)
