# Attribution and provenance

This repository is an adaptation of [satyamshivam13/Customer_Churn_Prediction](https://github.com/satyamshivam13/Customer_Churn_Prediction), organized around a standalone Kaggle experiment and the completed outputs supplied for Bank Customer Churn Prediction Using Machine Learning.

- The upstream MIT license is retained unchanged in `LICENSE`.
- `churn_preprocessing.py` retains the feature-engineering transformer from the completed Kaggle workflow and its serialized import path.
- `notebooks/Customer_Churn_Kaggle.ipynb` is the standalone notebook used for that workflow.
- `artifacts/kaggle_run` preserves the supplied outputs without substituting upstream or newly trained results. Python caches are omitted; a hash manifest is added.
- `bank_churn` provides a modular local implementation of that workflow, input checks, verification utilities, and separate output directories for new runs.
- The dataset CSV is the exact supplied input, with provenance to the Kaggle dataset referenced by the uploaded source project. The MIT software license does not establish a new dataset license.

The original application's FastAPI, Docker, and monitoring components are outside the requested training-and-evaluation repository. No deployment or monitoring results are claimed.
