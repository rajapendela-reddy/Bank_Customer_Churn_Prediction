# Dataset and schema

`raw/Churn_Modelling.csv` is the exact input CSV from the uploaded project.

SHA-256:

```text
ee2a5c9d8daf3aed0d91c4d883a84ebb77d271447bebc0f0c80aafb0f288ceab
```

The source project references [Bank Customer Churn Prediction on Kaggle](https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction). The software's MIT license does not replace the dataset publisher's terms.

| Column | Role | Meaning |
|---|---|---|
| RowNumber | Excluded | Row identifier |
| CustomerId | Split/prediction traceability; excluded from model | Customer identifier |
| Surname | Excluded | Customer surname |
| CreditScore | Numeric input | Customer credit score |
| Geography | Categorical input | Country; observed France, Germany, Spain |
| Gender | Categorical input | Gender category; observed Female, Male |
| Age | Numeric input | Customer age |
| Tenure | Numeric input | Customer tenure |
| Balance | Numeric input | Account balance |
| NumOfProducts | Numeric input | Number of bank products |
| HasCrCard | Numeric input | Credit-card indicator |
| IsActiveMember | Numeric input | Active-member indicator |
| EstimatedSalary | Numeric input | Estimated salary |
| Exited | Target | 0 = stayed; 1 = exited |

There are no observation dates in this CSV. No churn prediction horizon is established by the provided data.
