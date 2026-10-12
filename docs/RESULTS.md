# Interpretation of the completed Kaggle run

All metrics in this document describe the supplied run at threshold **0.5**. They are preserved separately from any later training runs.

## Test confusion matrix

| Actual / predicted | Stayed | Exited |
|---|---:|---:|
| Stayed | 1,541 | 52 |
| Exited | 228 | 179 |

- Correct classifications: 1,720 / 2,000 = **86.0%**.
- Churn precision: 179 / (179 + 52) = **77.49%**.
- Churn recall: 179 / (179 + 228) = **43.98%**.
- Churn F1: **0.5611285**.
- ROC-AUC: **0.8614963**.
- Always-stayed accuracy: 1,593 / 2,000 = **79.65%**.
- Accuracy improvement over that baseline: **6.35 percentage points**.

The classifier flags 231 customers, of whom 179 actually exited in the benchmark. It misses 228 churners (56.02% of the churners). High accuracy does not imply high churn recall.

ROC-AUC summarizes ranking across thresholds. The reported precision/recall/F1 instead describe the fixed threshold of 0.5. The notebook did not tune this threshold.

## Validation selection

Random Forest validation AUC is 0.8741071, XGBoost is 0.8732403, and Logistic Regression is 0.8591579. Random Forest wins the specified selection rule by a small margin over XGBoost. XGBoost has the higher validation recall and F1. No significance test or confidence interval is present, and no test comparison across all three models is claimed.

## Feature importance

The original Random Forest produces 24 encoded features. The leading entries are:

| Encoded feature | Approximate importance |
|---|---:|
| Age | 0.1755 |
| NumOfProducts | 0.1381 |
| AgeGroup: Senior | 0.1094 |
| EngagementScore | 0.1079 |
| HighProductRisk | 0.0917 |

These are the fitted forest's impurity-based importances. They are not changes in churn probability, causal effects, or independently measured customer-retention benefits. Engineered features are correlated with their source variables.

## Example predictions

The supplied example customers have probabilities 0.0695043150 and 0.9167237055, respectively, and predictions No and Yes. Actual future outcomes were not supplied.

## Supported project summary

Compared Logistic Regression, Random Forest, and XGBoost on a 10,000-customer benchmark using a stratified 60:20:20 split. Trained on 6,000 records and selected Random Forest by validation ROC-AUC. Achieved 86.0% accuracy and 0.8615 ROC-AUC on 2,000 test records, with a reusable preprocessing pipeline and five engineered features.

Claims of improved retention, financial savings, production deployment, SMOTE, threshold optimization, or statistical superiority are not supported by these artifacts.
