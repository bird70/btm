# Implementation Plan: Hybrid Segmentation Ensemble

**Branch**: `021-hybrid-segmentation` | **Date**: 2026-04-05 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/021-hybrid-segmentation/spec.md`

## Summary

Augment the existing RF/MLP Kaggle pipeline with a low-effort segmentation branch that is practical in notebook workflows. The plan benchmarks exactly three approaches (HF SegFormer fine-tuned, HF SegFormer + CRF, local DeepLabV3+), extracts per-location class and confidence features, and stacks these with RF/MLP outputs via multinomial logistic regression. Promotion is gated by validation quality (>=2pp weighted F1 lift) and minority-class safety (non-decreasing SGAM recall).

## Technical Context

**Language/Version**: Python >=3.11 (project baseline from `pyproject.toml`)  
**Primary Dependencies**: numpy, scipy, pandas, scikit-learn, pyyaml, torch, transformers, datasets, evaluate, pydensecrf, rasterio; tooling via kaggle CLI, huggingface_hub CLI, gh CLI  
**Storage**: Flat files in repo and artifacts: CSV (predictions/submissions), YAML (configs), JSON (metrics/provenance), model checkpoints under `artifacts/` or notebook output  
**Testing**: pytest for pipeline and utility tests; targeted integration checks for CLI workflow and artifact schema  
**Target Platform**: Local Python CLI and Kaggle notebook runtime (Linux GPU optional), plus GitHub workflow via gh CLI  
**Project Type**: Python monorepo with CLI-first ML pipeline (`python -m benthic_model.cli`)  
**Performance Goals**: End-to-end benchmark of 3 candidates in one working session; each single candidate training run completes within notebook/session budget (<2 hours per candidate baseline run)  
**Constraints**: Keep RF/MLP as baseline, no full pipeline port to notebook, no panoptic/GNN/custom architecture, reproducible with venv and command docs  
**Scale/Scope**: 590 train samples, 98 test samples, 5 habitat classes; one feature branch with docs/contracts/planning artifacts and follow-on implementation tasks

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **I. TDD** — Implementation tasks will start by adding failing tests for mask generation, candidate comparison reporting, and stacker gating before production code changes.
- [x] **II. Code Quality** — Python linting/test gates remain via ruff and pytest; no bypass is introduced.
- [x] **III. Performance** — Raster-facing logic remains vectorized (NumPy/SciPy/rasterio pathways); segmentation adds notebook model inference but does not replace block-safe core processing.
- [x] **IV. Scientific Accuracy** — Segmentation and CRF are established methods; benchmark reporting includes references and explicit baseline comparison rather than unsupported claims.
- [x] **V. Platform Portability** — Core data prep/stacking will remain pure Python in existing project modules/scripts; notebook runtime is an execution environment, not a platform lock-in.
- [x] **VI. Simplicity** — Scope is limited to 3 explicit approaches and one default meta-learner, avoiding speculative abstractions.

**Post-Design Re-check**: PASS. Phase 1 artifacts (research/data model/contracts/quickstart) keep the same guardrails and do not introduce constitution violations.

## Project Structure

### Documentation (this feature)

```text
specs/021-hybrid-segmentation/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── hybrid-segmentation-cli.md
└── tasks.md
```

### Source Code (repository root)

```text
src/benthic_model/
├── cli.py
├── features/
├── models/
├── evaluation/
├── inference/
└── submission/

configs/
├── *.yaml

scripts/
├── experiment_*.py

artifacts/
├── runs/
└── predictions/

tests/
├── unit/
├── integration/
└── contract/
```

**Structure Decision**: Use the existing monorepo and CLI pipeline structure. No new top-level package is introduced. Segmentation integration is expressed as pipeline-compatible artifacts and stacker inputs, preserving current command flow and reproducibility.

## Phase 0: Research Outcomes

See [research.md](research.md). All prior clarifications are resolved with concrete decisions for approach set, mask policy, imbalance handling, stacker default, and promotion gate.

## Phase 1: Design Outcomes

- Data entities and lifecycle are specified in [data-model.md](data-model.md).
- Interface and artifact contracts are defined in [contracts/hybrid-segmentation-cli.md](contracts/hybrid-segmentation-cli.md).
- Reproducible local/Kaggle/HF/GitHub workflow is documented in [quickstart.md](quickstart.md).

## Complexity Tracking

No constitution violations requiring justification.
