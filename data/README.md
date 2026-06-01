# Data

This directory contains the competition data files used by the notebooks.

## Source

- Competition: [FSI AIxData Challenge 2024](https://dacon.io/en/competitions/official/236297/overview/description)
- Organizer platform: DACON

The competition describes the data as fully synthetic financial transaction data. The files are included here for reproducibility within the portfolio repository.

## Expected Files

```text
data/
  train.csv
  test.csv
  sample_submission.csv
```

## Notes

- `train.csv` contains the `Fraud_Type` target.
- `test.csv` is used for final inference.
- `sample_submission.csv` defines the required submission format.
- Fold-specific and generated intermediate files are intentionally excluded from the public repository because they can be recreated from the notebooks.
