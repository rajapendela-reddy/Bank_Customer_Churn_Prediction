# Predicting customers

Edit `new_customers.csv`, then run from the repository root:

```bash
python -m bank_churn.predict --input examples/new_customers.csv --output predictions/new_customers.csv
```

Supply all ten raw input columns: `CreditScore`, `Geography`, `Gender`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`.

Column order is flexible. Numeric cells may be missing and will use the saved training medians. Unknown text categories are ignored by the fitted encoder. Invalid numeric strings, infinite values, missing required columns, and salary -1 are rejected. Extra fields, such as a customer reference, are kept in the output but not used as model predictors. Do not supply `Exited` as an input feature.

The output adds `churn_probability` and `predicted_churn` (Yes when probability is at least 0.5). This score has not been probability-calibrated and does not specify a future time horizon.

The included examples reproduce the two example records from the Kaggle experiment; their true outcomes are unknown.
