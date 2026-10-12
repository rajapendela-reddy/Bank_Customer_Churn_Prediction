# Experiment specification

## Data and split

Input: `Churn_Modelling.csv`, 10,000 rows, 14 columns. Exclude `RowNumber`, `CustomerId`, and `Surname` from predictors; keep `Exited` only as the target. Use the 10 raw features listed in `bank_churn/config.py`.

Two `train_test_split` calls use `random_state=42` and stratify on `Exited`:

1. Keep 60% for training and put 40% into a holdout partition.
2. Divide that holdout equally into validation and test.

Result: 6,000 training, 2,000 validation, 2,000 test. Compare `CustomerId` across partitions to rule out customer overlap. No resampling is performed.

## Preprocessing

The five engineered features are created by the original root-level `ChurnFeatureBuilder`. It is copied unchanged to retain compatibility with the supplied serialized pipelines. Group labels follow pandas' right-closed bin semantics:

- Tenure: `[0,3]`, `(3,6]`, `(6,9]`, `(9,11]`, labelled `0-3 years`, `3-6 years`, `6-9 years`, `9+ years`.
- Age: `[0,30]`, `(30,45]`, `(45,60]`, `(60,100]`, labelled `Young`, `Middle`, `Senior`, `Elderly`.

The labels are inherited from the existing workflow; exact boundaries above take precedence over informal label wording. Values outside the bins become missing and are handled by categorical imputation. `BalanceToSalaryRatio`, `HighProductRisk`, and `EngagementScore` use the formulas documented in the README.

Each classifier receives an independently created pipeline. Numeric fields: median imputation followed by standard scaling. Categorical fields: most-frequent imputation followed by dense one-hot encoding with unknown categories ignored. No preprocessor is fitted on validation or test data.

The repository CLI adds input checks for non-text categories and the undefined ratio denominator at salary -1; these reject malformed inputs without changing the supplied dataset or its transformations.

## Estimators

All unspecified parameters use library defaults from the installed runtime.

| Estimator | Explicit parameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `random_state=42` |
| Random Forest | `n_estimators=100`, `max_depth=10`, `min_samples_split=20`, `min_samples_leaf=10`, `n_jobs=2`, `random_state=42` |
| XGBoost | `n_estimators=100`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="logloss"`, `tree_method="hist"`, `n_jobs=2`, `random_state=42` |

Model selection: highest validation ROC-AUC. Decision threshold: `P(Exited=1) >= 0.5`. Only the selected model is evaluated on test, and that fitted pipeline is saved without a second fit.

The CLI uses stable sorting for exact validation-AUC ties, preserving estimator order. The provided run has no tie, so this does not change its selection.

## Files and runtime

`artifacts/kaggle_run` contains the unmodified outputs supplied from the completed Kaggle run, plus a provenance hash manifest. The archive's Python caches are omitted. The source notebook is also included with unexecuted cells.

`bank_churn` reorganizes the same computational workflow into local modules; new runs record their own runtime and data hash and do not overwrite original evidence. The original Kaggle Python and package versions are retained in its metadata. Retraining under the tested local dependency versions is a new run; it is not presented as the original run.

## Verification scope

`python -m bank_churn.verify --check-model` checks dataset bytes, split assignments and order, labels, prediction thresholds, independently recalculated metrics, saved-model predictions, example outputs, and original artifact hashes. It does not retrain the original models or establish performance on a different bank or time period.
