# Submission Artifacts

Submission files in this directory must conform to the Kaggle contract:

- Header exactly `ID,class`
- Exactly one row per test ID
- Unique IDs
- Class values restricted to training vocabulary

Use the CLI command:

`python -m benthic_model.cli make-submission --predictions <path> --sample-submission data/sample_submission.csv --output submissions/submission.csv`
