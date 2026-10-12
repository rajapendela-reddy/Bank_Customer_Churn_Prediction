# Upload the repository to GitHub

Suggested repository name: **bank-customer-churn-prediction**.

Suggested description:

> Bank customer churn classification with Logistic Regression, Random Forest, and XGBoost; includes verified Kaggle results, saved pipelines, and reproducible local training.

Suggested topics: `machine-learning`, `customer-churn`, `scikit-learn`, `random-forest`, `xgboost`, `python`, `kaggle`.

## Browser upload

1. Extract `Bank_Customer_Churn_Prediction_GitHub.zip`.
2. Create a new GitHub repository with the suggested name. Leave the automatic README, license, and gitignore options unchecked, because these files are already included.
3. Use **Add file → Upload files** and upload the **contents** of `Bank_Customer_Churn_Prediction`, so `README.md` is at the repository root. Upload the extracted files, not the ZIP as the entire project.
4. Commit the upload. Check that the README images render and the source, data, artifacts, notebook, documentation, and tests appear.
5. Ensure hidden files such as `.gitignore` and `.github/workflows/checks.yml` are included. Use the Git route if the browser upload omits them.

The repository does not contain a Git remote or `.git` directory. No external repository has been created or published automatically.

## Git upload

From the extracted project folder, run:

```bash
git init
git add .
git commit -m "Add reproducible bank customer churn project and verified Kaggle results"
git branch -M main
```

Copy the exact remote URL from the new repository's GitHub page, add it with `git remote add origin`, and push with `git push -u origin main`.

## Before publishing changes

```bash
python -m bank_churn.verify --check-model
python -m pytest -q
```

The original experiment is intentionally tracked. Generated runs and new predictions are ignored by Git; explicitly add a selected new run only if you intend to publish it. Keep dataset attribution and the original software license.
