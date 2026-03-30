from __future__ import annotations

from collections.abc import Iterable

from sklearn.metrics import f1_score


def weighted_f1(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    return float(f1_score(list(y_true), list(y_pred), average="weighted", zero_division=0))


def per_class_f1(
    y_true: Iterable[str], y_pred: Iterable[str], labels: list[str] | None = None
) -> dict[str, float]:
    true_values = list(y_true)
    pred_values = list(y_pred)

    if labels is None:
        labels = sorted(set(true_values) | set(pred_values))

    scores = f1_score(
        true_values,
        pred_values,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {label: float(score) for label, score in zip(labels, scores, strict=True)}
