<!--
Sync Impact Report
- Version change: template -> 1.0.0
- Modified principles:
	- Template Principle 1 -> I. Code Quality Is Release Quality
	- Template Principle 2 -> II. Model Performance Is Measured, Not Assumed
	- Template Principle 3 -> III. User Experience Consistency Is Mandatory
	- Template Principle 4 -> IV. Reproducibility and Traceability by Default
	- Template Principle 5 -> V. Small, Reviewable, Verifiable Changes
- Added sections:
	- Delivery Constraints
	- Development Workflow & Quality Gates
- Removed sections:
	- None
- Templates requiring updates:
	- ✅ updated: .specify/templates/plan-template.md
	- ✅ updated: .specify/templates/spec-template.md
	- ✅ updated: .specify/templates/tasks-template.md
	- ⚠ pending: .specify/templates/commands/*.md (directory not present in this repository)
- Deferred TODOs:
	- None
-->

# Benthic Terrain Model Kaggle Constitution

## Core Principles

### I. Code Quality Is Release Quality
All production changes MUST pass static analysis, formatting, and tests in CI before merge. Every pull
request MUST include clear intent, impacted modules, and regression risk notes. New behavior MUST include
or update automated tests at the smallest useful scope (unit first, then integration where cross-module
behavior exists). Rationale: quality controls reduce defect escape and keep model iteration safe.

### II. Model Performance Is Measured, Not Assumed
Every model-affecting change MUST define expected metric movement and evaluation dataset scope before
implementation. Changes MUST report baseline and candidate metrics side-by-side with reproducible commands
and random seeds when applicable. A model update MUST NOT be merged when it degrades primary competition
metrics beyond documented tolerance without explicit approval and rollback notes. Rationale: measurable
performance discipline prevents accidental regressions and supports evidence-based iteration.

### III. User Experience Consistency Is Mandatory
Any user-facing output (CLI messages, notebooks, reports, visualizations, documentation snippets) MUST use
consistent terminology, units, and ordering conventions across the repository. New UX surfaces MUST include
an acceptance checklist covering readability, actionability, and error clarity. Breaking UX changes MUST be
announced in quickstart or usage docs in the same pull request. Rationale: consistency lowers cognitive load
and improves trust in model outputs.

### IV. Reproducibility and Traceability by Default
Training, inference, and evaluation workflows MUST be reproducible from committed artifacts, including data
split references, feature generation steps, and dependency versions. Every experiment promoted to shared
artifacts MUST include provenance metadata (code revision, config, seed, and timestamp). Rationale:
traceability enables debugging, fair comparison, and reliable collaboration.

### V. Small, Reviewable, Verifiable Changes
Work MUST be delivered in small increments that can be reviewed and validated independently. Large changes
MUST be split by concern (data prep, modeling, evaluation, UX/reporting) with explicit dependency notes.
When a shortcut is taken for speed, a follow-up task with owner and due milestone MUST be recorded.
Rationale: smaller deltas reduce review risk and speed up reliable delivery.

## Delivery Constraints

- Python-based workflows MUST pin critical package versions for deterministic behavior.
- Evaluation scripts MUST emit machine-readable metric summaries alongside human-readable output.
- Any new external dependency MUST include justification and maintenance impact in the relevant plan.
- Documentation for setup and run commands MUST remain executable as written.

## Development Workflow & Quality Gates

1. Specification MUST define testable functional requirements, explicit success criteria, and user-impact
	 expectations for changed workflows.
2. Planning MUST include constitution checks for code quality, model performance, and UX consistency, with
	 measurable gates.
3. Tasks MUST include explicit quality tasks (lint/tests), performance validation tasks (baseline vs
	 candidate), and UX consistency validation tasks when output surfaces change.
4. Before merge, reviewers MUST verify evidence for all applicable gates and reject unverifiable claims.

## Governance

This constitution supersedes local conventions when conflicts occur. Amendments require: (1) a written
change proposal, (2) impact analysis on templates and active workflows, and (3) explicit approval from
repository maintainers. Semantic versioning policy applies to this document: MAJOR for incompatible
principle or governance changes, MINOR for new principles or materially expanded sections, PATCH for
clarifications with no behavior change. Compliance review is mandatory at plan approval and pull request
review; any exception MUST document scope, rationale, owner, and expiry.

**Version**: 1.0.0 | **Ratified**: 2026-03-24 | **Last Amended**: 2026-03-24
