import numpy as np

from benthic_model.evaluation.cv import (
    build_spatial_fold_assignments,
    iter_spatial_blocked_folds,
    iter_stratified_random_folds,
)


def test_spatial_fold_assignments_cover_all_samples() -> None:
    x = np.linspace(0, 10, 20)
    y = np.linspace(0, 20, 20)

    folds = build_spatial_fold_assignments(x=x, y=y, n_splits=5, spatial_bins=4, random_state=7)

    assert len(folds) == 20
    assert set(folds.tolist()).issubset({0, 1, 2, 3, 4})


def test_iter_spatial_blocked_folds_returns_disjoint_indices() -> None:
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)

    for train_idx, test_idx in iter_spatial_blocked_folds(x=x, y=y, n_splits=3, random_state=7):
        assert len(np.intersect1d(train_idx, test_idx)) == 0


def test_iter_stratified_random_folds_produces_k_splits() -> None:
    labels = ["A"] * 12 + ["B"] * 12

    splits = list(iter_stratified_random_folds(labels=labels, n_splits=4, random_state=13))

    assert len(splits) == 4
    for train_idx, test_idx in splits:
        assert len(train_idx) + len(test_idx) == len(labels)
