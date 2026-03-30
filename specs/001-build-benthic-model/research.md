# Phase 0 Research: Benthic Habitat Classification Pipeline

## Decision 1: Primary modeling family
- Decision: Use gradient-boosted trees (XGBoost) as the primary candidate model and Random Forest as baseline.
- Rationale: Tree ensembles handle mixed engineered geospatial features well, are robust with moderate tabular sizes, and optimize effectively against weighted F1.
- Alternatives considered: Logistic regression (too linear for terrain/acoustic interactions), SVM RBF (harder to scale/tune for iterative geospatial CV), deep CNN from rasters (high complexity relative to current data setup and timeline).

## Decision 2: Validation protocol for model selection
- Decision: Use spatial blocked k-fold as authoritative validation; run stratified random k-fold as secondary diagnostic only.
- Rationale: Geospatial autocorrelation can inflate non-spatial CV scores; blocked folds provide more realistic generalization estimates.
- Alternatives considered: Single holdout (high variance, unstable), stratified-only CV (leakage risk), leave-one-region-out (requires explicit region definitions not yet guaranteed in inputs).

## Decision 3: Feature extraction strategy from GeoTIFF
- Decision: Extract per-point bathymetry/backscatter values and augment with neighborhood statistics (windowed mean/std/min/max), simple gradients, and interaction features.
- Rationale: This captures local terrain context while remaining reproducible and computationally tractable.
- Alternatives considered: Raw raster patch CNN features (complex pipeline and data volume demands), extensive texture families (possible later if baseline/candidate plateau).

## Decision 4: Experiment reproducibility protocol
- Decision: Persist experiment metadata for every promoted run: code revision, config hash/path, seed, fold setup, metric summary path, and generated artifacts.
- Rationale: Constitution requires traceability and deterministic reruns within tolerance.
- Alternatives considered: Notebook-only ad hoc notes (insufficient traceability), lightweight metrics-only logging (missing provenance fields).

## Decision 5: User-facing interface contract
- Decision: Provide a scriptable CLI entrypoint (`train`, `evaluate`, `predict`, `make-submission`) plus notebooks for exploration.
- Rationale: CLI improves repeatability and CI compatibility; notebooks support local experimentation.
- Alternatives considered: Notebook-only workflow (poor automation/reproducibility), full web UI (out of scope and unnecessary for competition execution).

## Decision 6: Python environment and notebook workflow
- Decision: Standardize on Python 3.12 in project-local `.venv`, with Jupyter installed into the same environment and kernels bound to it.
- Rationale: Matches user environment preference and reduces dependency drift between scripts and notebooks.
- Alternatives considered: Global Python installation (non-reproducible), conda/pixi migration (possible future improvement, unnecessary for this phase).

## Decision 7: Submission validation guardrails
- Decision: Enforce submission contract checks: exact headers (`ID,class`), full ID coverage, unique IDs, class values constrained to training-label vocabulary.
- Rationale: Prevents invalid competition uploads and avoids avoidable leaderboard failures.
- Alternatives considered: Minimal write-only CSV generation (high formatting and data integrity risk).
