"""Reproduce the notebook workflow without deployment; save a separate new run."""
import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

from .config import DEFAULT_DATA, FEATURES, ROOT, SEED, TARGET
from .data import load_training_data, split_data
from .metrics import calculate_metrics
from .models import build_pipelines
from .plots import plot_training_data, plot_test, save_importance
from .predict import predict_frame


def train(data_path=DEFAULT_DATA, output=None, seed=SEED):
    frame, quality, data_hash = load_training_data(data_path)
    training, validation, test = split_data(frame, seed)
    if output is None:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f_UTC")
        output = ROOT / "artifacts/runs" / run_id
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output folder is not empty: {output}. Choose a new folder to preserve previous runs.")
    output.mkdir(parents=True, exist_ok=True)
    parts = [("train", training), ("validation", validation), ("test", test)]
    summary = pd.DataFrame([{"split": name, "rows": len(part), "churn_rate": part[TARGET].mean()}
                            for name, part in parts])
    summary.to_csv(output / "split_summary.csv", index=False)
    pd.concat([part[["CustomerId"]].assign(split=name) for name, part in parts],
              ignore_index=True).to_csv(output / "split_assignments.csv", index=False)
    (output / "data_quality.json").write_text(json.dumps(quality, indent=2))
    plot_training_data(training, output)
    pipelines = build_pipelines(seed)
    rows = []
    for name, pipeline in pipelines.items():
        print(f"Training: {name}", flush=True)
        pipeline.fit(training[FEATURES], training[TARGET])
        probabilities = pipeline.predict_proba(validation[FEATURES])[:, 1]
        rows.append({"model": name, **calculate_metrics(validation[TARGET], probabilities)})
        joblib.dump(pipeline, output / f"{name}_pipeline.joblib")
    comparison = pd.DataFrame(rows).sort_values("roc_auc", ascending=False, kind="stable")
    comparison.to_csv(output / "validation_model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best_pipeline = pipelines[best_name]
    # Test data is only scored after validation model selection.
    probability = best_pipeline.predict_proba(test[FEATURES])[:, 1]
    metrics = {"model": best_name, **calculate_metrics(test[TARGET], probability)}
    pd.DataFrame([metrics]).to_csv(output / "selected_model_test_metrics.csv", index=False)
    pd.DataFrame({"CustomerId": test["CustomerId"].to_numpy(),
                  "actual_churn": test[TARGET].to_numpy(),
                  "predicted_churn": (probability >= 0.5).astype(int),
                  "churn_probability": probability}).to_csv(output / "test_predictions.csv", index=False)
    plot_test(test[TARGET], probability, best_name, output)
    save_importance(pipelines["random_forest"], output)
    joblib.dump(best_pipeline, output / "best_churn_pipeline.joblib")
    # Same transformer module as the original joblib artifacts.
    (output / "churn_preprocessing.py").write_bytes((ROOT / "churn_preprocessing.py").read_bytes())
    metadata = {"selected_model": best_name, "selection_metric": "validation ROC-AUC",
                "decision_threshold": 0.5, "seed": seed,
                "split_fraction": {"train": 0.6, "validation": 0.2, "test": 0.2},
                "features": FEATURES, "source": str(Path(data_path).resolve()),
                "data_sha256": data_hash, "test_metrics": metrics,
                "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                             "numpy": np.__version__, "scikit-learn": sklearn.__version__,
                             "xgboost": xgboost.__version__, "joblib": joblib.__version__}}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    examples = pd.read_csv(ROOT / "examples/new_customers.csv")
    predict_frame(best_pipeline, examples).to_csv(output / "new_customer_predictions.csv", index=False)
    print(comparison.to_string(index=False))
    print("Selected by validation ROC-AUC:", best_name)
    print("Test metrics:", json.dumps(metrics, indent=2))
    print("Saved run:", output.resolve())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    train(args.data, args.output, args.seed)


if __name__ == "__main__":
    main()
