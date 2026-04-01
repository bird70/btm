# Tasks: Pipeline CV Improvement Investigation

**Input**: Design documents from `/specs/019-pipeline-cv-improvement/`
**Prerequisites**: plan.md ✓, spec.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

**Tests**: Not explicitly requested. Existing 236-test suite covers pipeline components — no new library code is introduced, so no new unit tests are written. Validation is by running experiments and observing CV outputs.

**Organization**: Tasks are grouped by user story. US1 (experiment sweep), US2 (seed variance quantification), and US3 (seed ensemble) can be implemented independently; US3 depends on US2 artifacts (prediction files).

---

## Phase 1: Setup

**Purpose**: Create YAML configurations and experiment script skeleton needed by all user stories

- [X] T001 Create `configs/lgbm-btm-tuned.yaml` — LightGBM with BTM-10 features, tuned for small dataset (fewer estimators, larger learning rate)
- [X] T002 [P] Create `configs/lgbm-btm-300.yaml` — LightGBM variant: n_estimators=300, learning_rate=0.05
- [X] T003 [P] Create `configs/lgbm-btm-ensemble.yaml` — RF+LightGBM soft-vote ensemble with BTM-10 features
- [X] T004 Create `scripts/experiment_v14_seed_ensemble.py` — skeleton script for majority-vote ensemble (reads predictions, writes output). Acceptance criterion: `--dry-run` flag loads 5 CSV paths, prints file count, exits 0.
- [X] T004a [P] Write unit test in `tests/unit/test_experiment_v14.py` covering majority-vote logic: unanimous 5/5 result, split 3/2 result, tiebreaker 2/2/1 using highest-CV seed
- [X] T004b [P] Create `docs/run-019-pipeline-cv-improvement.md` with placeholder section headers: §Summary, §RF Experiments, §LightGBM Experiments, §Feature Additions (FR-003), §Feature Flags (FR-004), §Seed Variance, §Seed Ensemble, §Submission Decision

**Checkpoint**: All configs are valid YAML parseable by `PipelineConfig.from_yaml`. Skeleton script passes `--dry-run`. Unit test for majority-vote logic is written and fails (TDD). Report file exists with section stubs.

---

## Phase 2: Foundational

**Purpose**: Confirm R04 baseline is reproducible on the current branch (required reference for all user stories)

**⚠️ CRITICAL**: No user story work can begin until the baseline CV score is confirmed

- [X] T005 Run R04 baseline: `benthic_model.cli train --config configs/rf-btm-fine.yaml --train-csv data/train_btm.csv --seed 42` and verify CV ≈ 0.8024 in `artifacts/runs/{run_id}/metrics.json`
- [X] T006 Run R04 predict: generate test predictions for the T005 run_id into `artifacts/predictions/{run_id}_test_predictions.csv`
- [X] T007 Run R04 make-submission: write `data/submission_v14_r04_baseline.csv` — establishes the submission format reference

**Checkpoint**: T005 CV within ±0.005 of 0.8024. Predictions and submission CSV exist and have 98 rows.

---

## Phase 3: User Story 1 — Systematic CV Improvement Investigation (Priority: P1) 🎯 MVP

**Goal**: Evaluate LightGBM and RF+LightGBM ensemble against the R04 baseline with statistical awareness of the seed noise floor

**Independent Test**: Each experiment produces a metrics.json. Results table shows CV ± comparison to R04. Any improvement claiming to be meaningful must exceed the 0.035 noise floor (2× seed std).

### RF Hyperparameter Sweep (consolidate prior findings)

- [X] T008 [P] [US1] Run S2 variant: `configs/rf-btm-s2-500trees.yaml` — CV=0.8001 (run candidate-20260401041240) if not already on this branch — record CV in experiment results table in `docs/run-019-pipeline-cv-improvement.md`
- [X] T009 [P] [US1] Run S9 variant: `configs/rf-btm-s9-log2.yaml` — CV=0.8024 (run candidate-20260401041256) — record CV in experiment results table

### LightGBM Experiments

- [X] T010 [US1] Run LightGBM tuned: `benthic_model.cli train --config configs/lgbm-btm-tuned.yaml --train-csv data/train_btm.csv --seed 42 --run-type candidate` — record CV and per-class F1
- [X] T011 [P] [US1] Run LightGBM 300: `benthic_model.cli train --config configs/lgbm-btm-300.yaml --train-csv data/train_btm.csv --seed 42 --run-type candidate` — record CV
- [X] T012 [P] [US1] Run RF+LightGBM ensemble: `benthic_model.cli train --config configs/lgbm-btm-ensemble.yaml --train-csv data/train_btm.csv --seed 42 --run-type candidate` — record CV

### LightGBM Multi-Seed Validation (if T010 or T011 shows promising CV)

- [X] T013 [US1] SKIPPED — no LightGBM variant achieved CV > 0.835 (lgbm-tuned=0.7976, lgbm-300=0.7959) (R04 mean + 2× std), run that config with seeds 123, 456, 789, 2026 to validate the improvement is not noise
- [X] T014 [US1] SKIPPED — T013 was skipped; all LightGBM variants below noise floor into `artifacts/predictions/` (depends on T013 completing — run_id is unknown until T013 identifies the winner)

### Evaluation

- [X] T015a [P] [US1] Document FR-003 small-window BTM findings from `research.md` §RT-3 into `docs/run-019-pipeline-cv-improvement.md` §Feature Additions — summarise all S5a–S5d results and confirm none exceed noise floor
- [X] T015b [P] [US1] Document FR-004 feature flag findings from `research.md` §RT-3 into `docs/run-019-pipeline-cv-improvement.md` §Feature Flags — summarise S1 (eco) and S6 (z-scores) results
- [X] T015 [US1] Populate US1 results table in `docs/run-019-pipeline-cv-improvement.md` — all CV scores, delta vs R04, noise-floor flag, recommendation

**Checkpoint**: Results table complete. Best model identified. Noise-floor analysis applied — only candidates with CV > 0.8374 (R04 seed=42 CV 0.8024 + noise floor 0.035) proceed to submission consideration.

---

## Phase 4: User Story 2 — Seed Variance Quantification (Priority: P2)

**Goal**: Measure CV variance across ≥5 seeds for R04 and (if promising) the best LightGBM config

**Independent Test**: Report shows mean, std, min, max across 5 seeds for R04. Statistical noise floor is computed and documented.

### Seed Runs (R04 — mostly completed from branch 018)

- [X] T016 Verify 5 seed runs for R04 already exist in `artifacts/runs/` (seeds 42, 123, 456, 789, 2026 with rf-btm-fine.yaml + train_btm.csv)
- [X] T017 [P] If any seed runs missing, execute: `benthic_model.cli train --config configs/rf-btm-fine.yaml --train-csv data/train_btm.csv --seed <missing_seed> --run-type candidate`

### Seed Runs (Best LightGBM — if applicable)

- [X] T018 [US2] SKIPPED — no LightGBM cleared noise floor (seed 42 already done in T010/T011)

### Variance Report

- [X] T019 [US2] Compute seed variance summary query `artifacts/runs/*/metrics.json` for all R04 seed runs, compute mean, std, min, max weighted_f1 — write to `docs/run-019-pipeline-cv-improvement.md` §Seed Variance
- [X] T020 [P] [US2] Generate test predictions for all 5 R04 seed runs (if not already in `artifacts/predictions/` from branch 018)

**Checkpoint**: Variance table populated. Noise floor (2× std) quantified. Document confirms prior finding: std ≈ 0.017, noise floor ≈ 0.035.

---

## Phase 5: User Story 3 — Seed Ensemble for Prediction Stability (Priority: P3)

**Goal**: Majority-vote ensemble across 5 seed predictions; compare to R04 single-seed

**Independent Test**: Ensemble CSV has 98 rows. Vote confidence distribution reported. Delta vs R04 single-seed reported.

**Prerequisite**: T020 complete (all 5 seed prediction files present)

- [X] T021 [US3] Implement majority-vote logic in `scripts/experiment_v14_seed_ensemble.py`: load 5 CSVs, compute per-row mode, write `data/submission_v14_seed_ensemble.csv` (depends on T004a unit test turning green; re-runs branch-018 ensemble for 019-branch traceability — expected result: 97/98 unanimous, ensemble = R04 predictions)
- [X] T022 [US3] Run `scripts/experiment_v14_seed_ensemble.py` — verify output has 98 rows, valid class values
- [X] T023 [P] [US3] Compare ensemble predictions to R04 (T007 submission): report how many of 98 samples changed, vote confidence distribution
- [X] T024 [US3] Verify ensemble against R04 predictions — prior finding shows 97/98 unanimous; if any new divergence, investigate
- [X] T025 [P] [US3] Write ensemble + comparison results to `docs/run-019-pipeline-cv-improvement.md` §Seed Ensemble

**Checkpoint**: Ensemble submission CSV written. Comparison report shows prediction stability analysis. Confirms whether ensemble differs from single-seed.

---

## Phase 6: Polish & Documentation

**Purpose**: Finalize submission candidate, document all findings, commit and push

- [X] T026 [P] Determine final submission candidate: pick best-validated config (CV > noise floor required, or R04 if nothing clears bar) — document justification in `docs/run-019-pipeline-cv-improvement.md` §Submission Decision
- [X] T027 Run `benthic_model.cli make-submission` for chosen candidate — write final `data/submission_v14_final.csv`
- [X] T028 [P] Update `docs/runsheet-hybrid-kaggle.md` with row for 019 experiment with row for 019 experiment — include CV, Kaggle slot budget note
- [X] T029 [P] Run full test suite: `.venv\Scripts\python.exe -m pytest tests/ -m "not arcgis and not qgis"` — confirm 0 failures
- [X] T030 Update `CHANGELOG` with entry for `019-pipeline-cv-improvement` with entry for `019-pipeline-cv-improvement` findings
- [ ] T031 `git add -A; git commit -m "019: pipeline CV improvement investigation complete"` — stage all artifacts, configs, scripts, docs, specs
- [ ] T032 `git push --set-upstream origin 019-pipeline-cv-improvement`
- [ ] T033 Create GitHub PR: base=main (or relevant integration branch), title "019: Pipeline CV Improvement Investigation", body from `docs/run-019-pipeline-cv-improvement.md` §Summary

**Checkpoint**: Branch pushed. PR created. All experiment artifacts committed.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — start immediately
- **Foundational (Phase 2)**: Depends on Phase 1 completion — BLOCKS all user stories
- **US1 (Phase 3)**: Depends on Phase 2 (baseline confirmed). LightGBM experiments are independent of each other [P]
- **US2 (Phase 4)**: Depends on Phase 2. Seed runs mostly pre-exist from branch 018
- **US3 (Phase 5)**: Depends on US2 T020 (prediction files complete)
- **Polish (Phase 6)**: Depends on US1, US2, US3 completing

### User Story Dependencies

- **US1 (P1)**: Starts after Phase 2. Independent of US2/US3
- **US2 (P2)**: Starts after Phase 2. Independent of US1 (uses pre-existing seed runs)
- **US3 (P3)**: Starts after US2 T020 (prediction files needed). Independent of US1

### Parallel Opportunities

**Within Phase 1**: T002, T003 can run in parallel with T001/T004  
**Within Phase 3**: T008, T009, T011, T012 [all P] can run in parallel  
**Within Phase 4**: T017, T020 [P] can run in parallel with T016  
**Within Phase 5**: T023, T025 [P] can run after T022  
**Within Phase 6**: T028, T029, T030 [all P] can run in parallel

---

## Parallel Example: User Story 1

```bash
# Run these in separate terminals simultaneously:
# Terminal 1
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m benthic_model.cli train `
  --config configs/lgbm-btm-tuned.yaml --train-csv data/train_btm.csv `
  --bathymetry-tif data/MBES/bathymetry.tif --backscatter-tif data/MBES/backscatter.tif `
  --run-type candidate --seed 42

# Terminal 2
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m benthic_model.cli train `
  --config configs/lgbm-btm-300.yaml --train-csv data/train_btm.csv `
  --bathymetry-tif data/MBES/bathymetry.tif --backscatter-tif data/MBES/backscatter.tif `
  --run-type candidate --seed 42

# Terminal 3
$env:PYTHONPATH="src"; .venv\Scripts\python.exe -m benthic_model.cli train `
  --config configs/lgbm-btm-ensemble.yaml --train-csv data/train_btm.csv `
  --bathymetry-tif data/MBES/bathymetry.tif --backscatter-tif data/MBES/backscatter.tif `
  --run-type candidate --seed 42
```

---

## Implementation Strategy

**MVP = User Story 1 (Phase 3)**

The minimum deliverable is: run LightGBM experiments, apply noise-floor analysis, conclude whether R04 is beatable. If no model exceeds the noise floor, US2 and US3 provide supporting evidence that the investigation was thorough.

**Incremental delivery order**:

1. Phase 1 + Phase 2 (setup + baseline confirmation) — ~10 minutes
2. Phase 3 US1 (LightGBM experiments) — ~20 minutes (3 parallel CLI runs)
3. Phase 4 US2 (seed variance — mostly pre-done) — ~5 minutes verification
4. Phase 5 US3 (seed ensemble) — ~5 minutes
5. Phase 6 (polish, commit, PR) — ~15 minutes

**Total estimated runtime**: ~55 minutes

---

## Format Validation

All tasks follow: `- [ ] [TaskID] [P?] [Story?] Description with file path`

- Tasks T001–T033 are all checkboxes with sequential numeric IDs ✓
- Parallelizable tasks marked with [P] ✓
- User story tasks marked with [US1], [US2], [US3] ✓
- All tasks reference concrete file paths ✓
- No [P] on tasks with unresolved dependencies ✓
- Setup/Foundational tasks have no story label ✓
