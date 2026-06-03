import pandas as pd
from sklearn.ensemble import IsolationForest


def score_facilities(model: IsolationForest, facilities_df: pd.DataFrame) -> pd.Series:
    """Return anomaly scores indexed by facility_id, values 0.0–1.0 (higher = more anomalous)."""
    features = facilities_df[["tests_per_month", "cartridge_availability"]].values
    raw_scores = model.decision_function(features)
    # decision_function: higher = more normal; invert and normalise to [0,1]
    normalised = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    return pd.Series(normalised, index=facilities_df["facility_id"])
