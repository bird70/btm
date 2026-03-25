<!--
  Sync Impact Report
  ==================
  Version change: 1.0.0 → 1.1.0 (MINOR: new principle added; tech stack materially rewritten)
  Modified principles:
    - III. Performance First — removed ArcPy-specific compression mandate; made platform-neutral
    - V. Simplicity & Maintainability → renamed V. Platform Portability (new content)
    - VI. Simplicity & Maintainability — new section (content migrated + updated from old V)
  Added sections:
    - V. Platform Portability (new principle)
    - VI. Simplicity & Maintainability (renumbered from V)
  Removed sections: N/A
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ updated (Constitution Check gates)
    - .specify/templates/spec-template.md ✅ no changes required
    - .specify/templates/tasks-template.md ✅ no changes required
  Follow-up TODOs: None.

  ---

  Version change: 1.1.0 → 1.2.0 (MINOR: materially expanded Principle V governance guidance
    — new exception clause for legacy-migration priority inversion; plus PATCH fix for
    internal ArcGIS version floor inconsistency)
  Modified principles:
    - V. Platform Portability — added "Legacy Migration Features" exception clause
      documenting when ArcGIS adapter may be scoped higher priority than QGIS adapter.
  Modified sections:
    - Technology Stack §ArcGIS Pro path — corrected target version from ≥ 3.4 to ≥ 3.2
      (was inconsistent with Principle V which already stated ≥ 3.2; ≥ 3.2 is the
      more conservative/inclusive floor and aligns with spec/plan/tasks).
  Templates requiring updates:
    - .specify/templates/plan-template.md ✅ no changes required (gates unchanged)
    - .specify/templates/spec-template.md ✅ no changes required
    - .specify/templates/tasks-template.md ✅ no changes required
  Follow-up TODOs: None.
-->

# Benthic Terrain Modeler (BTM) Constitution

## Core Principles

### I. Test-Driven Development (NON-NEGOTIABLE)

Tests MUST be written before implementation code. No feature or bug-fix may be merged without
accompanying automated tests. The Red-Green-Refactor cycle is strictly enforced:

- Write a failing test that captures the requirement.
- Implement the minimum code to make the test pass.
- Refactor while keeping all tests green.
- Unit tests MUST cover individual scripts and functions in isolation.
- Integration tests MUST cover tool execution end-to-end using real raster data under `tests/data/`.
- Test coverage MUST NOT regress with any change.

### II. Code Quality Standards

All code MUST meet the following quality bar before merging:

- PEP 8 compliance is enforced via a linter (flake8 or ruff); CI MUST fail on lint errors.
- Functions and classes MUST have clear, descriptive names; public APIs MUST have docstrings.
- Magic numbers and hardcoded paths are forbidden; use named constants or configuration.
- Dead code MUST be removed; commented-out code is NOT permitted in committed files.
- Code review is REQUIRED for every change; no self-merge is allowed.

### III. Performance First

BTM operates on large geospatial rasters; performance is a first-class concern:

- Block-based (chunked) processing MUST be used for raster operations to bound memory usage.
- NumPy/SciPy vectorised operations MUST be preferred over Python-level pixel loops.
- Performance benchmarks MUST be recorded for any tool that processes rasters above 100 MB.
- New algorithms MUST document time and space complexity in their docstrings.
- Output rasters MUST use lossless compression regardless of the backend used to write them.

### IV. Scientific Accuracy

Results must be geospatially and mathematically correct:

- All geomorphometric algorithms (BPI, VRM, slope, arc-chord ratio, etc.) MUST trace their
  implementation to a cited peer-reviewed source or the original BTM specification.
- Any deviation from the reference implementation MUST be documented and justified.
- Classification boundaries (zone and structure) MUST be validated against the canonical
  Fagatele Bay reference dataset included in `tests/data/`.

### V. Platform Portability

The modernization MUST decouple geospatial algorithm logic from any single GIS platform:

- Every algorithm MUST be implemented as a pure-Python core function with no hard dependency on
  arcpy, QGIS, or any other GIS runtime. Core logic lives in `btm/core/` (or equivalent).
- Platform-specific wrappers (ArcGIS geoprocessing tool, QGIS processing provider, CLI entry
  point) MUST delegate to the core function; they MUST NOT re-implement algorithm logic.
- **Priority order for new or rewritten tools**:
  1. Standalone Python script (no GIS runtime required; use rasterio/GDAL for I/O).
  2. QGIS Processing provider (where integration adds clear user value).
  3. ArcGIS Pro Python Toolbox (`.pyt`) targeting ArcGIS Pro ≥ 3.2 / Python 3.
  4. ArcGIS-only fallback: permitted ONLY when the algorithm is provably impossible to
     implement without arcpy (e.g., deep ModelBuilder integration). MUST be documented.
- **Exception — Legacy Migration Features**: When a feature is specifically designed to
  restore compatibility for an existing ArcGIS-only user base migrating from a broken
  toolbox version, the ArcGIS Pro adapter (item 3 above) MAY be scoped at a higher feature
  priority than the QGIS provider (item 2 above) within that feature's delivery plan.
  This exception MUST be explicitly documented in the feature spec with a clear rationale.
  _(Applied in `001-btm-portable-core`: the existing BTM 3.0 user base is ArcGIS-centric;
  US3/ArcGIS Pro is scoped as P3 and US4/QGIS as P4, justified by migration continuity
  for current users — see spec.md §US3 rationale.)_
- Tools SHOULD provide an optional CLI interface (`python -m btm.<tool> --help`) so they
  run without any GUI.
- Windows-specific code is permitted only inside ArcGIS wrappers; all core code and
  QGIS/standalone paths MUST be cross-platform.

### VI. Simplicity & Maintainability

- YAGNI: implement only what is required; no speculative abstractions.
- Each module MUST have a single, well-defined responsibility.
- ArcGIS toolbox XML parameter definitions MUST remain in sync with their Python tool class.
- QGIS provider metadata MUST remain in sync with the underlying algorithm signature.
- Submodule dependencies (e.g., `datatype`) MUST be pinned to a specific commit SHA.

## Technology Stack & Constraints

### Core (required for all paths)

- **Language**: Python 3.11+ (minimum Python 3.11 for QGIS/standalone compatibility)
- **Raster I/O (non-ArcGIS paths)**: rasterio + GDAL; must be available in the standard
  `pip`-installable dependency set
- **Numerical computing**: NumPy, SciPy — used in all paths
- **Classification I/O**: openpyxl / csv (stdlib); no xlrd dependency on new code
- **Testing Framework**: pytest; suite MUST pass via `pytest` from the repository root
  without any GIS runtime installed (tests that require arcpy MUST be skipped via a marker
  when arcpy is absent)

### ArcGIS Pro path (optional integration)

- **Target version**: ArcGIS Pro ≥ 3.2 (Python 3.x conda environment)
- **Interface**: Python Toolbox (`.pyt`) only; the legacy `.esriaddin` format is frozen
  and MUST NOT be extended as part of the modernization
- **Extension**: Spatial Analyst required for tools that use arcpy spatial analyst functions;
  MUST degrade gracefully (clear error message) when the extension is unavailable

### QGIS path (preferred for open-source users)

- **Target version**: QGIS LTR ≥ 3.34
- **Interface**: QGIS Processing provider registered via a plugin or standalone provider
- **Dependency**: PyQGIS (qgis.core); isolated from core algorithm logic

### Standalone / CLI path

- **Interface**: `python -m btm.<tool>` entry point using argparse or click
- **Dependencies**: rasterio, NumPy, SciPy only — no GIS runtime needed
- **Platform**: Cross-platform (Windows, macOS, Linux)

### Performance Targets

- All tools MUST complete within a reasonable duration on datasets up to 1 GB
- The block-based processor is MANDATORY for rasters exceeding available RAM

## Development Workflow

- All work MUST be done in a feature branch; direct commits to `main` are forbidden.
- Every branch MUST include a spec (`spec.md`), plan (`plan.md`), and passing tests before merge.
- CI MUST run in order: lint check → unit tests (no GIS runtime) → integration tests.
- Integration tests that require arcpy or QGIS MUST be gated behind environment markers
  (`@pytest.mark.arcgis`, `@pytest.mark.qgis`) and MUST NOT block CI on machines
  without those runtimes.
- A CHANGELOG entry MUST be added for every user-visible change, following the pattern
  established in `CHANGELOG`.
- When modernizing an existing tool, the legacy ArcGIS implementation MUST remain in the
  `legacy/` tree until the replacement passes all existing integration tests.
- The `.esriaddin` format is frozen; new GUI work MUST target the ArcGIS Pro `.pyt`
  interface or a QGIS plugin.

## Governance

This constitution supersedes all other informal practices or conventions.
Amendments REQUIRE:

1. A documented rationale added to the Sync Impact Report comment at the top of this file.
2. An incremented version number following semantic versioning (MAJOR.MINOR.PATCH):
   - MAJOR: backward-incompatible principle removal or redefinition.
   - MINOR: new principle or section added, or materially expanded guidance.
   - PATCH: clarifications, wording, or typo fixes.
3. All dependent templates (`plan-template.md`, `spec-template.md`, `tasks-template.md`)
   reviewed and updated as needed.
4. Confirmation that no existing tests are invalidated; if they are, a migration plan
   MUST accompany the amendment.

All PRs and code reviews MUST verify compliance with these principles. Violations MUST be
flagged and resolved before merge. This constitution MUST be re-reviewed annually or after
any major platform upgrade (e.g., new ArcGIS Pro or Python version).

**Version**: 1.2.0 | **Ratified**: 2026-03-25 | **Last Amended**: 2026-03-25
