# Final UX Consistency Checklist

## Terminology
- [X] Metrics use canonical names: `weighted_f1`, `per_class_f1`
- [X] Class labels match training vocabulary formatting
- [X] Fold scheme names are `spatial_blocked` or `stratified_random`

## Interface Output
- [X] CLI messages include failing input path when reporting errors
- [X] Reports use consistent decimal precision and metric names
- [X] Submission docs use exact header spelling: `ID,class`

## Workflow
- [X] Quickstart examples match CLI argument names
- [X] Notebook examples align with current pipeline commands
- [X] Artifact paths shown in docs match repository layout

## Validation Notes
- Candidate run was blocked by threshold policy and baseline was selected for release submission.
- Submission contract checks passed (row count, uniqueness, header format, class vocabulary).
