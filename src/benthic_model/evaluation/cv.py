from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def _quantile_bin(values: np.ndarray, bins: int) -> np.ndarray:
    # qcut can reduce bin count when duplicate values occur; labels stay deterministic.
    binned = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    return np.asarray(binned, dtype=int)


def build_spatial_fold_assignments(
    x: Iterable[float],
    y: Iterable[float],
    n_splits: int = 5,
    spatial_bins: int = 4,
    random_state: int = 42,
) -> np.ndarray:
    x_arr = np.asarray(list(x), dtype=float)
    y_arr = np.asarray(list(y), dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError("x and y coordinate arrays must have the same length.")
    if len(x_arr) < n_splits:
        raise ValueError("Number of samples must be at least n_splits.")

    x_bin = _quantile_bin(x_arr, spatial_bins)
    y_bin = _quantile_bin(y_arr, spatial_bins)
    block_ids = (x_bin * (y_bin.max() + 1)) + y_bin

    unique_blocks = np.unique(block_ids)
    rng = np.random.default_rng(seed=random_state)
    shuffled_blocks = rng.permutation(unique_blocks)

    block_to_fold = {
        int(block): int(index % n_splits) for index, block in enumerate(shuffled_blocks)
    }
    folds = np.array([block_to_fold[int(block)] for block in block_ids], dtype=int)
    return folds


def iter_spatial_blocked_folds(
    x: Iterable[float],
    y: Iterable[float],
    n_splits: int = 5,
    spatial_bins: int = 4,
    random_state: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    folds = build_spatial_fold_assignments(
        x=x,
        y=y,
        n_splits=n_splits,
        spatial_bins=spatial_bins,
        random_state=random_state,
    )
    indices = np.arange(len(folds))
    for fold_id in range(n_splits):
        test_idx = indices[folds == fold_id]
        train_idx = indices[folds != fold_id]
        if len(test_idx) == 0:
            continue
        yield train_idx, test_idx


def iter_stratified_random_folds(
    labels: Iterable[str], n_splits: int = 5, random_state: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(list(labels))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    X = np.zeros((len(y), 1), dtype=float)
    yield from splitter.split(X, y)
