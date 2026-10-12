"""Build a fresh independent preprocessor for each estimator."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# This import path is deliberately preserved for the original serialized models.
from churn_preprocessing import ChurnFeatureBuilder
from .config import CATEGORICAL, NUMERIC


def build_preprocessor():
    columns = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         CATEGORICAL),
    ])
    return Pipeline([("features", ChurnFeatureBuilder()), ("columns", columns)])
