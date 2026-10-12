"""Audit saved results against their prediction files and the source dataset."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from .config import DEFAULT_DATA, FEATURES, ORIGINAL_RUN, TARGET
from .data import load_training_data, split_data
from .metrics import calculate_metrics


def verify_results(run=ORIGINAL_RUN, data_path=DEFAULT_DATA, check_model=False):
    run = Path(run)
    frame, quality, data_hash = load_training_data(data_path)
    metadata = json.loads((run / "run_metadata.json").read_text())
    if data_hash != metadata["data_sha256"]:
        raise ValueError("Dataset bytes do not match the run's SHA-256 hash.")
    if metadata["decision_threshold"] != 0.5:
        raise ValueError("This workflow expects the recorded threshold of 0.5.")
    parts = split_data(frame, int(metadata["seed"]))
    names = ["train", "validation", "test"]
    expected_assignments = pd.concat([part[["CustomerId"]].assign(split=name)
                                     for name, part in zip(names, parts)], ignore_index=True)
    assignments = pd.read_csv(run / "split_assignments.csv")
    pd.testing.assert_frame_equal(assignments, expected_assignments, check_dtype=False)
    if assignments["CustomerId"].duplicated().any():
        raise ValueError("Customer overlap in the split manifest.")
    summary = pd.read_csv(run / "split_summary.csv")
    for i, part in enumerate(parts):
        if summary.iloc[i]["split"] != names[i] or summary.iloc[i]["rows"] != len(part):
            raise ValueError("Split counts are inconsistent.")
        np.testing.assert_allclose(summary.iloc[i]["churn_rate"], part[TARGET].mean(), atol=1e-12, rtol=0)
    recorded_quality = json.loads((run / "data_quality.json").read_text())
    if quality != recorded_quality:
        raise ValueError("Data-quality report differs from the dataset.")
    comparison = pd.read_csv(run / "validation_model_comparison.csv")
    if set(comparison["model"]) != {"logistic_regression", "random_forest", "xgboost"}:
        raise ValueError("Expected validation results for all three models.")
    selected = comparison.sort_values("roc_auc", ascending=False, kind="stable").iloc[0]["model"]
    if selected != metadata["selected_model"]:
        raise ValueError("Selected model does not match validation ROC-AUC selection.")
    predictions = pd.read_csv(run / "test_predictions.csv")
    test = parts[2]
    np.testing.assert_array_equal(predictions["CustomerId"], test["CustomerId"])
    np.testing.assert_array_equal(predictions["actual_churn"], test[TARGET])
    probability = predictions["churn_probability"].to_numpy()
    computed = calculate_metrics(predictions["actual_churn"], probability)
    np.testing.assert_array_equal(predictions["predicted_churn"], (probability >= 0.5).astype(int))
    saved = pd.read_csv(run / "selected_model_test_metrics.csv").iloc[0]
    if saved["model"] != selected:
        raise ValueError("Test report model differs from selected model.")
    for key, value in computed.items():
        np.testing.assert_allclose(saved[key], value, atol=1e-12, rtol=0)
        np.testing.assert_allclose(metadata["test_metrics"][key], value, atol=1e-12, rtol=0)
    importance = pd.read_csv(run / "random_forest_feature_importance.csv")
    if importance["feature"].duplicated().any() or not np.isfinite(importance["importance"]).all():
        raise ValueError("Invalid feature-importance records.")
    if (importance["importance"] < 0).any():
        raise ValueError("Negative tree importance.")
    np.testing.assert_allclose(importance["importance"].sum(), 1.0, atol=1e-12, rtol=0)
    if check_model:
        from .predict import load_pipeline, predict_frame
        pipeline = load_pipeline(run / "best_churn_pipeline.joblib")
        np.testing.assert_allclose(pipeline.predict_proba(test[FEATURES])[:, 1], probability, atol=1e-12, rtol=0)
        examples = pd.read_csv(run / "new_customer_predictions.csv")
        actual = predict_frame(pipeline, examples.drop(columns=["churn_probability", "predicted_churn"]))
        np.testing.assert_allclose(actual["churn_probability"], examples["churn_probability"], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(actual["predicted_churn"], examples["predicted_churn"])
    manifest = run / "artifact_manifest.json"
    if manifest.exists():
        entries = json.loads(manifest.read_text())["files"]
        for entry in entries:
            candidate = (run / entry["path"]).resolve()
            if not candidate.is_relative_to(run.resolve()):
                raise ValueError("Invalid artifact path.")
            actual_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
            if actual_hash != entry["sha256"]:
                raise ValueError(f"Original artifact changed: {entry['path']}")
    result = {"dataset_sha256": data_hash, "rows": len(frame), "selected_model": selected,
              "test_metrics": computed,
              "confusion_matrix_actual_rows_predicted_columns": confusion_matrix(test[TARGET], probability >= 0.5).tolist(),
              "always_stayed_baseline_accuracy": float((test[TARGET] == 0).mean()),
              "saved_model_predictions_checked": check_model}
    print("Verification passed.")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ORIGINAL_RUN)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--check-model", action="store_true", help="Also reproduce predictions with the saved pipeline.")
    args = parser.parse_args()
    verify_results(args.run, args.data, args.check_model)


if __name__ == "__main__":
    main()
