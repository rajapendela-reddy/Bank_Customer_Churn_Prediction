"""Constants matching the completed Kaggle experiment."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/Churn_Modelling.csv"
ORIGINAL_RUN = ROOT / "artifacts/kaggle_run"
SEED = 42
THRESHOLD = 0.5
TARGET = "Exited"
BASE_NUMERIC = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "HasCrCard", "IsActiveMember", "EstimatedSalary"]
BASE_CATEGORICAL = ["Geography", "Gender"]
FEATURES = BASE_NUMERIC + BASE_CATEGORICAL
NUMERIC = BASE_NUMERIC + ["BalanceToSalaryRatio", "HighProductRisk", "EngagementScore"]
CATEGORICAL = BASE_CATEGORICAL + ["TenureGroup", "AgeGroup"]
REFERENCE_DATA_SHA256 = "ee2a5c9d8daf3aed0d91c4d883a84ebb77d271447bebc0f0c80aafb0f288ceab"
