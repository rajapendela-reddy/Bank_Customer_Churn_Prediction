"""Schema validation and deterministic stratified splitting."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import BASE_CATEGORICAL, BASE_NUMERIC, FEATURES, SEED, TARGET


def validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    result = frame[FEATURES].copy()
    for col in BASE_NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="raise")
    if np.isinf(result[BASE_NUMERIC].to_numpy(dtype=float)).any():
        raise ValueError("Numeric features contain infinite values.")
    if (result["EstimatedSalary"] == -1).any():
        raise ValueError("EstimatedSalary=-1 makes BalanceToSalaryRatio undefined.")
    for col in BASE_CATEGORICAL:
        result[col] = result[col].map(lambda v: v.strip() if isinstance(v, str) else v)
        result[col] = result[col].replace("", np.nan)
        result[col] = result[col].where(result[col].notna(), np.nan)
        values = result[col].dropna()
        if not values.map(lambda v: isinstance(v, str)).all():
            raise ValueError(f"{col} must contain text values or missing values.")
    return result


def load_training_data(path: Path):
    raw = Path(path).read_bytes()
    frame = pd.read_csv(path)
    required = FEATURES + [TARGET, "CustomerId"]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    quality = {"initial_rows": len(frame), "exact_duplicates": int(frame.duplicated().sum()),
               "missing_by_column": frame[required].isna().sum().astype(int).to_dict()}
    frame = frame.drop_duplicates().copy()
    if frame["CustomerId"].isna().any() or frame["CustomerId"].duplicated().any():
        raise ValueError("CustomerId must be present and unique.")
    frame[FEATURES] = validate_features(frame)
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise")
    if frame[TARGET].isna().any() or not frame[TARGET].isin([0, 1]).all():
        raise ValueError("Exited must contain only 0 or 1, without missing values.")
    frame[TARGET] = frame[TARGET].astype(int)
    if frame[TARGET].nunique() != 2:
        raise ValueError("Training requires both classes.")
    quality.update(cleaned_rows=len(frame), churn_rate=float(frame[TARGET].mean()))
    return frame, quality, hashlib.sha256(raw).hexdigest()


def split_data(frame: pd.DataFrame, seed: int = SEED):
    training, holdout = train_test_split(frame, test_size=0.4, stratify=frame[TARGET], random_state=seed)
    validation, test = train_test_split(holdout, test_size=0.5, stratify=holdout[TARGET], random_state=seed)
    ids = [set(part["CustomerId"]) for part in (training, validation, test)]
    if any(ids[a] & ids[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Customer overlap across splits.")
    return training, validation, test
