"""The three estimators and hyperparameters from the Kaggle notebook."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import SEED
from .preprocessing import build_preprocessor


def build_pipelines(seed=SEED):
    estimators = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10,
            min_samples_split=20, min_samples_leaf=10, n_jobs=2, random_state=seed),
        "xgboost": XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            tree_method="hist", n_jobs=2, random_state=seed),
    }
    return {name: Pipeline([("preprocessing", build_preprocessor()), ("model", estimator)])
            for name, estimator in estimators.items()}
