# Repository validation

Validated on **2026-10-12** in a local CPU environment using Python 3.12.14 and the package versions pinned in `requirements.txt`.

## Completed checks

- `python -m pytest -q`: **11 passed**.
- Original saved Random Forest: reproduced all 2,000 saved test probabilities and both example probabilities within absolute tolerance `1e-12`.
- Full refactored training run: trained all three classifiers using the exact supplied dataset and seed 42; selected Random Forest by validation ROC-AUC.
- Refactored Random Forest: reproduced the original test prediction CSV within absolute tolerance `1e-12`.
- Recalculated original and new Random Forest test metrics: accuracy **0.86**, precision **0.7748917749**, recall **0.4398034398**, F1 **0.5611285266**, ROC-AUC **0.8614963191**.
- Verified the exact CSV SHA-256, split membership and order, customer uniqueness, labels, decision threshold, and artifact hashes.
- Ran the prediction CLI successfully on `examples/new_customers.csv`.
- Checked notebook format, notebook cell syntax, Python source syntax, workflow YAML, and local Markdown links.

The test suite emitted one SciPy optimizer deprecation warning during Logistic Regression fitting; no test failed.

## Cross-version distinction

The tested local XGBoost version is **2.1.4**, whereas the supplied Kaggle metadata records **3.4.1**. The local XGBoost validation AUC was **0.8706338063**, versus **0.8732402823** in the original report. New-run XGBoost results are not substituted into the original experiment. The original XGBoost model file is retained, but loading it under a different XGBoost version has not been verified.

The original notebook was validated structurally; the standalone notebook was not re-executed inside Kaggle during repository preparation. GitHub Actions is configured but has not been run on a hosted GitHub repository. The modular local training, prediction, and verification workflow was executed successfully.
