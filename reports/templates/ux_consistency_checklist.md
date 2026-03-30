# UX Consistency Checklist

## Terminology
- [ ] Metrics use canonical names: `weighted_f1`, `per_class_f1`
- [ ] Class labels match training vocabulary formatting
- [ ] Fold scheme names are `spatial_blocked` or `stratified_random`

## Interface Output
- [ ] CLI messages include failing input path when reporting errors
- [ ] Reports use consistent decimal precision and metric names
- [ ] Submission docs use exact header spelling: `ID,class`

## Workflow
- [ ] Quickstart examples match CLI argument names
- [ ] Notebook examples align with current pipeline commands
- [ ] Artifact paths shown in docs match repository layout
