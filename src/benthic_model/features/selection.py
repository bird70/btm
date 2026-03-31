"""Permutation importance-based feature selection for benthic habitat models.

Reduces a high-dimensional multi-scale feature set to a compact subset
using a two-step approach:

1. Spearman correlation pre-filter: for any pair of features with
   |r| > ``corr_threshold``, the lower-variance feature is dropped to
   eliminate redundant multi-scale derivatives before the expensive
   permutation step.

2. Permutation importance: uses ``sklearn.inspection.permutation_importance``
   to estimate each feature's contribution to model accuracy.  Features with
   ``importance_mean ≤ importance_threshold`` are dropped.

References
----------
Breiman, L. (2001).
    Random forests.  *Machine Learning*, 45(1), 5–32.
    (Permutation importance originally proposed in this work.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

_log = logging.getLogger(__name__)


@dataclass
class FeatureSelectionResult:
    """Record of which features were retained after permutation importance.

    Attributes
    ----------
    selected_features:
        List of retained feature column names.
    dropped_features:
        List of feature column names that were dropped.
    importance_df:
        DataFrame with columns ``feature_name``, ``importance_mean``,
        ``importance_std``, ``selected``, ``rank``.
    """

    selected_features: list[str] = field(default_factory=list)
    dropped_features: list[str] = field(default_factory=list)
    importance_df: pd.DataFrame = field(default_factory=pd.DataFrame)


def select_by_permutation_importance(
    X: pd.DataFrame,
    y: pd.Series,
    model,
    n_repeats: int = 10,
    importance_threshold: float = 0.0,
    corr_threshold: float = 0.95,
) -> FeatureSelectionResult:
    """Select features via Spearman correlation pre-filter + permutation importance.

    Steps
    -----
    1. Compute Spearman correlation matrix.  For any pair (A, B) where
       |r| > ``corr_threshold``, drop the feature with lower variance
       (keep higher-variance to maximise retained information).
    2. Call ``sklearn.inspection.permutation_importance`` using the supplied
       ``model`` (which should already be fitted on ``X``).
    3. Drop all features where ``importance_mean ≤ importance_threshold``.
    4. Log retained and dropped features with their importance scores.

    Parameters
    ----------
    X:
        Feature DataFrame (samples × features).  The model must have been
        trained on this feature set (or a superset).
    y:
        Target series aligned with ``X``.
    model:
        Fitted estimator with a ``predict`` method compatible with
        ``sklearn.inspection.permutation_importance``.
    n_repeats:
        Number of permutation repeats (higher → more stable estimates).
    importance_threshold:
        Features with ``importance_mean ≤ this value`` are dropped.
        Default 0.0 drops only features that harm or do not help accuracy.
    corr_threshold:
        Absolute Spearman correlation threshold above which the lower-variance
        feature of a correlated pair is dropped (default 0.95).

    Returns
    -------
    FeatureSelectionResult

    References
    ----------
    Breiman (2001) — see module docstring.
    """
    from sklearn.inspection import permutation_importance

    feature_names = list(X.columns)

    # ------------------------------------------------------------------
    # Step 1: Spearman correlation pre-filter
    # ------------------------------------------------------------------
    corr_matrix = X.corr(method="spearman").abs()
    variances = X.var()

    to_drop_corr: set[str] = set()
    upper_tri = np.triu(corr_matrix.values, k=1)
    high_corr_pairs = np.argwhere(upper_tri > corr_threshold)

    for idx_a, idx_b in high_corr_pairs:
        feat_a = feature_names[idx_a]
        feat_b = feature_names[idx_b]
        if feat_a in to_drop_corr or feat_b in to_drop_corr:
            continue
        # Drop the lower-variance feature
        if variances[feat_a] >= variances[feat_b]:
            to_drop_corr.add(feat_b)
        else:
            to_drop_corr.add(feat_a)

    if to_drop_corr:
        _log.info(
            "Correlation pre-filter dropped %d features (|r| > %.2f): %s",
            len(to_drop_corr),
            corr_threshold,
            ", ".join(sorted(to_drop_corr)),
        )

    # ------------------------------------------------------------------
    # Step 2: Permutation importance on ALL features
    # (model was trained on X; permuting a subset would cause sklearn
    # validation errors — run on full X and post-filter)
    # ------------------------------------------------------------------
    result = permutation_importance(
        model,
        X,
        y,
        n_repeats=n_repeats,
        random_state=42,
    )

    imp_means = result.importances_mean
    imp_stds = result.importances_std

    # ------------------------------------------------------------------
    # Step 3: Threshold filtering + corr filter combined
    # ------------------------------------------------------------------
    selected: list[str] = []
    dropped_perm: list[str] = []
    for name, mean_imp in zip(feature_names, imp_means):
        if name in to_drop_corr:
            dropped_perm.append(name)
        elif mean_imp > importance_threshold:
            selected.append(name)
        else:
            dropped_perm.append(name)

    all_dropped = dropped_perm

    # ------------------------------------------------------------------
    # Step 4: Log results
    # ------------------------------------------------------------------
    for name, mean_imp, std_imp in zip(feature_names, imp_means, imp_stds):
        status = "KEEP" if name in selected else "DROP"
        _log.info(
            "  %s  %-40s  importance=%.4f ± %.4f",
            status,
            name,
            mean_imp,
            std_imp,
        )

    _log.info(
        "Feature selection: retained %d / %d; dropped %d (corr pre-filter or low importance)",
        len(selected),
        len(feature_names),
        len(all_dropped),
    )

    # ------------------------------------------------------------------
    # Build importance DataFrame
    # ------------------------------------------------------------------
    imp_df = pd.DataFrame(
        {
            "feature_name": feature_names,
            "importance_mean": imp_means.tolist(),
            "importance_std": imp_stds.tolist(),
            "selected": [n in selected for n in feature_names],
        }
    )
    imp_df["rank"] = (
        imp_df["importance_mean"]
        .rank(ascending=False, na_option="bottom")
        .astype(int)
    )

    return FeatureSelectionResult(
        selected_features=selected,
        dropped_features=all_dropped,
        importance_df=imp_df,
    )
