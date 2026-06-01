# Project Summary

## One-Line Summary

Built a fraud-type classification pipeline for FSI AIxData Challenge 2024 using CTGAN-based minority-class augmentation, stratified validation, under-sampling, and weighted ensemble modeling.

## Why This Project Matters

The task resembles a practical financial AI problem: the important classes are rare, the table contains mixed feature types, and local validation can be misleading if the split does not preserve the target distribution. The project therefore emphasized validation design and pipeline reliability rather than only model complexity.

## Core Decisions

| Decision | Reason |
| --- | --- |
| Use CTGAN for minority classes | Increase training signal for rare fraud types |
| Generate synthetic data by fold | Reduce validation leakage risk |
| Use stratified 5-fold validation | Preserve fraud-type distribution during validation |
| Test under-sampling ratios | Control majority-class dominance |
| Compare ensemble strategies | Improve stability across leaderboard splits |

## Main Result

- Official result: Public 28th / Private 22nd
- Retrospective result after fixing submission logic: potential Public 27th / Private 20th

## What I Would Improve Next

- Build a reusable training script instead of relying only on notebooks.
- Add a configuration file for model parameters and sampling ratios.
- Track experiments with fixed seeds, fold IDs, and score tables.
- Add automated checks for label mapping and submission ZIP structure.
