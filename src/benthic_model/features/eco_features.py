"""Ecology-informed in-pipeline feature transformations.

This module handles **training-time** transformations only:

1. ``btm_depth_zone`` — ordinal integer (1–4) representing 4 quantile-based
   depth bins computed from the training-set depth distribution.
2. ``btm_sgam_niche`` — binary indicator (0 or 1) encoding the joint
   ecological niche of *Amphibolis antarctica* seagrass: low fine-scale BPI
   AND low slope AND depth within the observed SGAM depth range in the
   training data.

Raster-level spatial derivatives (northness, eastness, max curvature,
complexity) are **not** computed here — they live in
``btm/features/extract.py`` (raster tier) and are pre-computed before
training via the ``btm-export-features`` CLI.

References
----------
Wilson, M. F. J., O'Connell, B., Brown, C., Guinan, J. C., & Grehan, A. J.
    (2007). Multiscale terrain analysis of multibeam bathymetry data for
    habitat mapping on the continental slope. *Marine Geodesy*, 30(1–2),
    3–35.

Sappington, J. M., Longshore, K. M., & Thompson, D. B. (2007).
    Quantifying landscape ruggedness for animal habitat analysis: A
    case study using bighorn sheep in the Mojave Desert, California.
    *The Journal of Wildlife Management*, 71(5), 1419–1426.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

# Number of quantile-based depth bins (ordinal labels 1 … N_DEPTH_BINS).
N_DEPTH_BINS: int = 4

# Class label for the SGAM seagrass class used when computing niche thresholds.
_SGAM_LABEL: str = "SGAM"

# Sentinel value stored in depth_bin_edges_ when not yet fitted.
_UNFITTED_SENTINEL: object = object()


class EcoFeatureTransformer:
    """Compute ecology-informed depth zone and SGAM niche features at training time.

    Follows the scikit-learn transformer API (``fit`` / ``transform`` /
    ``fit_transform``) so it can be embedded in the feature engineering
    pipeline alongside other transformers.

    Attributes
    ----------
    depth_bin_edges_ : list[float]
        5 edges defining the 4 quantile-based depth bins (set after ``fit``).
    p25_bpi_ : float
        25th percentile of ``btm_fine_bpi`` in the training set.
    p25_slope_ : float
        25th percentile of ``btm_slope`` in the training set.
    sgam_depth_min_ : float
        Minimum ``bathymetry`` value among SGAM-labelled training rows.
    sgam_depth_max_ : float
        Maximum ``bathymetry`` value among SGAM-labelled training rows.
    """

    def __init__(self) -> None:
        # Use sentinel so we can detect unfitted instances.
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "EcoFeatureTransformer":
        """Compute and store threshold statistics from the training DataFrame.

        Parameters
        ----------
        df:
            Training feature DataFrame.  Must contain columns
            ``bathymetry``, ``btm_fine_bpi``, and ``btm_slope``.
        y:
            Optional label series aligned to ``df``.  Used to derive the
            SGAM depth range (rows where ``y == 'SGAM'``).  When ``None``
            or when no SGAM rows exist, ``sgam_depth_min_`` and
            ``sgam_depth_max_`` are both set to 0.0.

        Returns
        -------
        self
        """
        depth = df["bathymetry"].to_numpy(dtype=float)
        bpi = df["btm_fine_bpi"].to_numpy(dtype=float)
        slope = df["btm_slope"].to_numpy(dtype=float)

        # Depth bins: 4 equal-frequency bins at p25/p50/p75 of training depth.
        percentiles = np.percentile(depth, [0, 25, 50, 75, 100])
        # Ensure strictly increasing edges (handles near-constant depth).
        edges: list[float] = list(percentiles)
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-9
        self.depth_bin_edges_: list[float] = edges

        # Percentile thresholds for SGAM niche indicator.
        self.p25_bpi_: float = float(np.percentile(bpi, 25))
        self.p25_slope_: float = float(np.percentile(slope, 25))

        # SGAM depth range.
        if y is not None:
            sgam_mask = (y == _SGAM_LABEL).to_numpy()
            if sgam_mask.any():
                sgam_depths = depth[sgam_mask]
                self.sgam_depth_min_: float = float(sgam_depths.min())
                self.sgam_depth_max_: float = float(sgam_depths.max())
            else:
                self.sgam_depth_min_ = 0.0
                self.sgam_depth_max_ = 0.0
        else:
            self.sgam_depth_min_ = 0.0
            self.sgam_depth_max_ = 0.0

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,  # kept for API symmetry; not used at transform time
    ) -> pd.DataFrame:
        """Add ``btm_depth_zone`` and ``btm_sgam_niche`` columns to ``df``.

        Parameters
        ----------
        df:
            Feature DataFrame containing at minimum ``bathymetry``,
            ``btm_fine_bpi``, and ``btm_slope``.

        Returns
        -------
        pd.DataFrame
            New DataFrame (copy of ``df``) with two additional columns:
            ``btm_depth_zone`` (int 1–4) and ``btm_sgam_niche`` (int 0/1).

        Raises
        ------
        sklearn.exceptions.NotFittedError
            When called before ``fit()``.
        """
        if not self._fitted:
            raise NotFittedError(
                "This EcoFeatureTransformer instance is not fitted yet. "
                "Call 'fit' before 'transform'."
            )

        out = df.copy()
        depth = out["bathymetry"].to_numpy(dtype=float)
        bpi = out["btm_fine_bpi"].to_numpy(dtype=float)
        slope = out["btm_slope"].to_numpy(dtype=float)

        # Assign ordinal depth zone (1 = shallowest bin, 4 = deepest bin).
        # np.searchsorted returns 0 for values below the first edge and
        # N for values above the last edge; we clamp to [1, N_DEPTH_BINS].
        edges = np.array(self.depth_bin_edges_)
        # Use right=False so a value exactly at a bin edge falls in the
        # *next* bin, consistent with pd.cut(right=True, include_lowest=True).
        zone_raw = np.searchsorted(edges[1:-1], depth, side="left") + 1
        # Clamp to valid range (handles values exactly at min/max edge).
        out["btm_depth_zone"] = np.clip(zone_raw, 1, N_DEPTH_BINS).astype(int)

        # SGAM niche indicator: 1 when BPI low, slope low, depth in range.
        niche = (
            (bpi < self.p25_bpi_)
            & (slope < self.p25_slope_)
            & (depth >= self.sgam_depth_min_)
            & (depth <= self.sgam_depth_max_)
        ).astype(int)
        out["btm_sgam_niche"] = niche

        return out

    def fit_transform(
        self,
        df: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> pd.DataFrame:
        """Fit on ``df`` and then transform it.

        Parameters
        ----------
        df:
            Training feature DataFrame.
        y:
            Optional label series for SGAM depth range derivation.

        Returns
        -------
        pd.DataFrame
            Transformed copy of ``df``.
        """
        return self.fit(df, y).transform(df)

    # ------------------------------------------------------------------
    # Serialisation (for eco_thresholds.json artifact)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise fitted parameters to a JSON-serialisable dict.

        Returns an empty dict when the transformer has not been fitted.
        Use ``from_dict({})`` and check for ``NotFittedError`` when
        reconstructing from an empty dict.
        """
        if not self._fitted:
            return {}
        return {
            "depth_bin_edges": self.depth_bin_edges_,
            "p25_bpi": self.p25_bpi_,
            "p25_slope": self.p25_slope_,
            "sgam_depth_min": self.sgam_depth_min_,
            "sgam_depth_max": self.sgam_depth_max_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EcoFeatureTransformer":
        """Reconstruct a fitted transformer from a serialised dict.

        Parameters
        ----------
        d:
            Dict produced by ``to_dict()``.  An empty dict returns an
            *unfitted* instance (``transform()`` will raise
            ``NotFittedError``).

        Returns
        -------
        EcoFeatureTransformer
        """
        t = cls()
        if not d:
            return t  # unfitted sentinel

        t.depth_bin_edges_ = list(d["depth_bin_edges"])
        t.p25_bpi_ = float(d["p25_bpi"])
        t.p25_slope_ = float(d["p25_slope"])
        t.sgam_depth_min_ = float(d["sgam_depth_min"])
        t.sgam_depth_max_ = float(d["sgam_depth_max"])
        t._fitted = True
        return t
