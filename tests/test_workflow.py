"""Regression checks for leakage, portability, and the provided run evidence."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn_preprocessing import ChurnFeatureBuilder
from bank_churn.config import DEFAULT_DATA, FEATURES, ORIGINAL_RUN, ROOT, TARGET
from bank_churn.data import load_training_data, split_data, validate_features
from bank_churn.predict import load_pipeline, predict_frame
from bank_churn.preprocessing import build_preprocessor
from bank_churn.verify import verify_results


@pytest.fixture(scope="module")
def dataset():
    return load_training_data(DEFAULT_DATA)[0]


def test_split_matches_the_original_run(dataset):
    parts = split_data(dataset)
    assert [len(p) for p in parts] == [6000, 2000, 2000]
    assert [int(p[TARGET].sum()) for p in parts] == [1222, 408, 407]
    result = verify_results()
    assert result["confusion_matrix_actual_rows_predicted_columns"] == [[1541, 52], [228, 179]]


def test_feature_boundaries_and_missing_values(dataset):
    raw = dataset[FEATURES].iloc[:4].copy()
    raw["Tenure"] = [0, 3, 6, 10]
    raw["Age"] = [30, 45, 60, 61]
    raw["NumOfProducts"] = [2, 3, np.nan, 4]
    before = raw.copy(deep=True)
    transformed = ChurnFeatureBuilder().transform(raw)
    pd.testing.assert_frame_equal(raw, before)
    assert transformed["TenureGroup"].tolist() == ["0-3 years", "0-3 years", "3-6 years", "9+ years"]
    assert transformed["AgeGroup"].tolist() == ["Young", "Middle", "Senior", "Elderly"]
    assert transformed["HighProductRisk"].iloc[0] == 0
    assert transformed["HighProductRisk"].iloc[1] == 1
    assert pd.isna(transformed["HighProductRisk"].iloc[2])
    np.testing.assert_allclose(transformed["EngagementScore"], raw["IsActiveMember"] + raw["NumOfProducts"] / 4)


def test_training_only_statistics_and_unseen_categories(dataset):
    training, _, test = split_data(dataset)
    prep = build_preprocessor().fit(training[FEATURES], training[TARGET])
    scaler = prep.named_steps["columns"].named_transformers_["numeric"].named_steps["scale"]
    assert int(scaler.n_samples_seen_) == 6000
    np.testing.assert_allclose(scaler.mean_[0], training["CreditScore"].mean())
    unseen = test[FEATURES].iloc[:2].copy()
    unseen["Geography"] = "Unseen country"
    unseen.loc[unseen.index[0], "Balance"] = np.nan
    matrix = prep.transform(unseen)
    assert matrix.shape == (2, 24) and np.isfinite(matrix).all()
    assert build_preprocessor().named_steps["columns"] is not prep.named_steps["columns"]


def test_saved_original_model_reproduces_test_and_example_predictions():
    verify_results(check_model=True)


def test_new_pipeline_serialization_roundtrip(dataset, tmp_path):
    training, _, test = split_data(dataset)
    pipeline = Pipeline([("preprocessing", build_preprocessor()),
                         ("model", LogisticRegression(max_iter=1000, random_state=42))])
    pipeline.fit(training[FEATURES], training[TARGET])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    np.testing.assert_allclose(joblib.load(path).predict_proba(test[FEATURES].iloc[:8]),
                               pipeline.predict_proba(test[FEATURES].iloc[:8]), atol=0, rtol=0)


@pytest.mark.parametrize("change", ["missing", "infinity", "invalid_numeric", "zero_denominator"])
def test_invalid_prediction_inputs_are_rejected(dataset, change):
    frame = dataset[FEATURES].iloc[:2].copy()
    if change == "missing": frame = frame.drop(columns="Age")
    elif change == "infinity": frame["Balance"] = np.inf
    elif change == "invalid_numeric": frame["Age"] = "not a number"
    else: frame["EstimatedSalary"] = -1
    with pytest.raises((ValueError, TypeError)):
        validate_features(frame)


def test_incompatible_sklearn_version_is_rejected_before_loading(tmp_path):
    (tmp_path / "run_metadata.json").write_text(json.dumps({"versions": {"scikit-learn": "0.0.0"}}))
    with pytest.raises(RuntimeError, match="scikit-learn"):
        load_pipeline(tmp_path / "not_loaded.joblib")


def test_empty_customers_are_rejected():
    with pytest.raises(ValueError, match="no customers"):
        predict_frame(None, pd.DataFrame(columns=FEATURES))
