# Football Predictor — Copilot Instructions

## Project
This is a football match result prediction ML engine focused ONLY on UEFA Champions League matches.

The project is being built from scratch.

## Current dataset
The master cleaned dataset is:

data/processed/ucl_matches_clean.csv

The chronological dataset is:

data/processed/ucl_matches_dated.csv

The temporal feature dataset is:

data/processed/ucl_features_v1.csv

Current temporal dataset:
- 7,366 matches
- 47 columns
- Target: result_after_90
- Classes:
  - HOME_WIN
  - DRAW
  - AWAY_WIN

## Critical data leakage rule

This project is extremely strict about temporal leakage.

For a match at timestamp T:
- Features may use information from matches strictly before T.
- A match's own result must never affect its features.
- Matches occurring at exactly the same timestamp must all calculate features from histories existing before that timestamp.
- Only after every feature row for timestamp T has been calculated may the completed matches at T update historical state.

Never introduce future information into historical features.

## Current temporal feature engine

src/features/temporal.py

The current build_features() implementation processes matches grouped by timestamp in two phases:

Phase 1:
Calculate all feature rows using pre-timestamp histories.

Phase 2:
Update team histories and recent forms using completed matches from that timestamp.

Do not revert this to a single-row update loop.

## Dataset ordering

Chronological ordering is required for temporal features.

Dates must be valid timestamps.

Do not silently treat NaT/missing timestamps as valid chronological information.

## Current feature groups

The current 47-column feature dataset contains:
- match metadata
- recent form
- overall historical performance
- home/away performance
- competition metadata
- target

Do not add features merely for complexity. Every new feature must have a clear predictive rationale and must obey the temporal leakage rule.

## ML methodology

Never use a random train/test split for the primary evaluation.

The project should use chronological train/validation/test splits.

The final test set must represent genuinely later matches than the training data.

Do not fit preprocessing, encoders, scalers, feature selectors, or models using future/test data.

## Engineering principles

- Prefer small, testable modules.
- Avoid duplicated feature-engineering logic.
- Keep training and inference feature definitions identical.
- Use pathlib and configuration rather than hardcoded machine-specific paths.
- Add tests for important data and leakage assumptions.
- Do not silently modify datasets.
- Preserve reproducibility with explicit random seeds where applicable.
- Explain potentially destructive changes before making them.

## Current development stage

Temporal feature generation v1 is complete.

Next:
1. Add automated temporal leakage tests.
2. Validate feature integrity.
3. Create chronological train/validation/test datasets.
4. Establish baseline ML models.
5. Evaluate using appropriate classification metrics.
6. Improve the feature engine only after establishing a baseline.
7. Build inference/prediction functionality.
8. Build API/UI later.

Do not jump directly to deployment.