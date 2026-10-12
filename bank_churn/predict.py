"""Local CSV predictions with the complete saved preprocessing/model pipeline."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

# Required when loading trusted models whose transformer was serialized at this path.
import churn_preprocessing  # noqa: F401
from .config import FEATURES, ORIGINAL_RUN, ROOT
from .data import validate_features


def load_pipeline(path):
    path = Path(path)
    metadata = path.parent / "run_metadata.json"
    if metadata.exists():
        recorded = json.loads(metadata.read_text()).get("versions", {}).get("scikit-learn")
        if recorded and recorded != sklearn.__version__:
            raise RuntimeError(f"This model was saved with scikit-learn {recorded}; installed version is "
                               f"{sklearn.__version__}. Install scikit-learn=={recorded} before loading it.")
    return joblib.load(path)


def predict_frame(pipeline, frame):
    if frame.empty:
        raise ValueError("Input CSV has no customers.")
    features = validate_features(frame)
    probabilities = pipeline.predict_proba(features[FEATURES])[:, 1]
    result = frame.copy()
    result["churn_probability"] = probabilities
    result["predicted_churn"] = np.where(probabilities >= 0.5, "Yes", "No")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "examples/new_customers.csv")
    parser.add_argument("--model", type=Path, default=ORIGINAL_RUN / "best_churn_pipeline.joblib")
    parser.add_argument("--output", type=Path, default=ROOT / "predictions/new_customer_predictions.csv")
    args = parser.parse_args()
    result = predict_frame(load_pipeline(args.model), pd.read_csv(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()
