"""Run once (after generate_data.py) to train and serialise models.

Usage:
    python scripts/train_models.py
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED, FACILITIES_CSV, CDST_CSV,
    ISOLATION_FOREST_PATH, QTABLE_PATH, ARTIFACTS_DIR,
    ORS_MATRIX_CACHE, N_FACILITIES, N_CDST, N_ACTIONS,
    ALPHA, BETA, GAMMA,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_EPISODES = 15_000
LR = 0.1            # Q-learning rate (α)
DISCOUNT = 0.0      # γ — single-step decision, no future state
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.9997


# ── Helpers ───────────────────────────────────────────────────────────────────

def _haversine_minutes(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
    speed_kmh: float = 45.0,
) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2)
    km = 2 * R * np.arcsin(np.sqrt(a))
    return km / speed_kmh * 60.0


def _build_travel_matrix(facilities_df: pd.DataFrame, cdst_df: pd.DataFrame) -> np.ndarray:
    if ORS_MATRIX_CACHE.exists():
        print("  Using cached ORS travel matrix.")
        return np.load(ORS_MATRIX_CACHE)
    print("  ORS cache not found — computing Haversine approximation.")
    matrix = np.zeros((N_FACILITIES, N_CDST))
    for i, fac in facilities_df.iterrows():
        for j, cdst in cdst_df.iterrows():
            matrix[i, j] = _haversine_minutes(fac["lat"], fac["lon"], cdst["lat"], cdst["lon"])
    return matrix


def _composite_weight(
    travel_min: float,
    queue_load: float,
    max_capacity: float,
    cartridge: float,
) -> float:
    norm_travel = min(travel_min / 600.0, 1.0)
    norm_queue = queue_load / max(max_capacity, 1.0)
    norm_cart_unavail = 1.0 - min(cartridge / 150.0, 1.0)
    return ALPHA * norm_travel + BETA * norm_queue + GAMMA * norm_cart_unavail


# ── Training functions ────────────────────────────────────────────────────────

def train_isolation_forest(facilities_df: pd.DataFrame) -> None:
    from sklearn.ensemble import IsolationForest
    features = facilities_df[["tests_per_month", "cartridge_availability"]].values
    model = IsolationForest(random_state=RANDOM_SEED, contamination=0.1)
    model.fit(features)
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ISOLATION_FOREST_PATH)
    print(f"  Isolation Forest → {ISOLATION_FOREST_PATH}")


def train_qtable(facilities_df: pd.DataFrame, cdst_df: pd.DataFrame) -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    travel_matrix = _build_travel_matrix(facilities_df, cdst_df)

    capacities = cdst_df["monthly_capacity"].values.astype(float)
    qtable = np.zeros((N_FACILITIES, N_ACTIONS))

    print(f"  Training Q-table: {N_EPISODES} episodes, ε {EPSILON_START}→{EPSILON_END}...")

    for ep in range(N_EPISODES):
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** ep))

        # Sample environment state: random facility + simulated CDST dynamics
        fac_idx = int(rng.integers(0, N_FACILITIES))
        cart_sim = rng.integers(0, 151, size=N_CDST).astype(float)
        queue_sim = np.array([
            rng.integers(0, int(capacities[j]) + 1) for j in range(N_CDST)
        ], dtype=float)

        # ε-greedy action selection
        if rng.random() < epsilon:
            action = int(rng.integers(0, N_ACTIONS))
        else:
            action = int(np.argmax(qtable[fac_idx]))

        # Reward: negative composite cost (lower cost = better)
        weight = _composite_weight(
            travel_min=float(travel_matrix[fac_idx, action]),
            queue_load=queue_sim[action],
            max_capacity=capacities[action],
            cartridge=cart_sim[action],
        )
        reward = -weight

        # Single-step Q-update: Q(s,a) ← Q(s,a) + α(r − Q(s,a))
        qtable[fac_idx, action] += LR * (reward - qtable[fac_idx, action])

        if (ep + 1) % 5_000 == 0:
            print(f"    Episode {ep+1:>6}/{N_EPISODES}  ε={epsilon:.4f}  "
                  f"mean_Q={qtable.mean():.4f}")

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    np.save(QTABLE_PATH, qtable)
    print(f"  Q-table → {QTABLE_PATH}  shape={qtable.shape}")
    _print_qtable_summary(qtable, facilities_df, cdst_df)


def _print_qtable_summary(
    qtable: np.ndarray,
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
) -> None:
    preferred = np.argmax(qtable, axis=1)
    cdst_ids = cdst_df["cdst_id"].tolist()
    counts = {c: int((preferred == i).sum()) for i, c in enumerate(cdst_ids)}
    print("  Learned routing preferences (facilities → preferred CDST):")
    for cdst_id, count in counts.items():
        name = cdst_df.loc[cdst_df["cdst_id"] == cdst_id, "cdst_name"].values[0]
        print(f"    {cdst_id} ({name}): {count} facilities")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not FACILITIES_CSV.exists():
        raise FileNotFoundError("Run scripts/generate_data.py first.")
    print("Loading data...")
    facilities_df = pd.read_csv(FACILITIES_CSV)
    cdst_df = pd.read_csv(CDST_CSV)

    print("Training Isolation Forest...")
    train_isolation_forest(facilities_df)

    print("Training Q-table...")
    train_qtable(facilities_df, cdst_df)

    print("Done.")
