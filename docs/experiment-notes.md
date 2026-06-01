# Experiment Notes

## Baseline

The initial baseline used tree-based classifiers after categorical encoding. This provided a quick reference point, but did not sufficiently address the severe target imbalance.

## CTGAN Augmentation

CTGAN was introduced to generate additional minority-class samples. The key implementation detail was to generate synthetic records from training folds only, then validate on untouched fold data.

## Under-Sampling

Under-sampling was tested to reduce the majority-class effect. The final direction used under-sampling as a control mechanism rather than a standalone solution.

## Model Comparison

The main classifiers were:

- XGBoost
- LightGBM
- CatBoost
- RandomForest baseline

Model quality was compared with fold-level F1, precision, recall, and accuracy tables.

## Ensemble Tests

Three ensemble styles were explored:

- Voting
- Stacking
- Weighted blending

Weighted blending was preferred because it was simple, interpretable, and stable enough for the final submission.

## Submission Check

The competition required a ZIP containing:

```text
clf_submission.csv
syn_submission.csv
```

The retrospective review showed that submission-file logic and label mapping were important enough to affect leaderboard rank, so the final portfolio version keeps the submission generation step explicit.
