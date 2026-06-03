import json
import logging

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import (
    ISOLATION_FOREST_PATH, QTABLE_PATH, MANIFEST_JSON, RANDOM_SEED,
    FACILITIES_CSV, CDST_CSV,
)

logger = logging.getLogger(__name__)


def load_models(
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
    travel_matrix: np.ndarray,
):
    """Load and validate all pre-trained artefacts. Raises on seed mismatch."""
    _validate_manifest()
    iso_forest = _load_isolation_forest()
    qtable = _load_qtable()
    shap_explainer = _build_shap_explainer(travel_matrix, facilities_df, cdst_df)
    return iso_forest, qtable, shap_explainer


def _validate_manifest():
    if not MANIFEST_JSON.exists():
        raise FileNotFoundError(
            f"Manifest not found at {MANIFEST_JSON}. Run scripts/train_models.py first."
        )
    manifest = json.loads(MANIFEST_JSON.read_text())
    if manifest.get("random_seed") != RANDOM_SEED:
        raise ValueError(
            f"Artefact seed {manifest['random_seed']} != config RANDOM_SEED {RANDOM_SEED}"
        )
    logger.info("Manifest validated: seed=%s", RANDOM_SEED)


def _load_isolation_forest():
    if not ISOLATION_FOREST_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {ISOLATION_FOREST_PATH}. Run scripts/train_models.py."
        )
    assert ISOLATION_FOREST_PATH.stem.endswith(f"seed{RANDOM_SEED}"), \
        f"Artefact seed mismatch: {ISOLATION_FOREST_PATH.stem}"
    return joblib.load(ISOLATION_FOREST_PATH)


def _load_qtable():
    if not QTABLE_PATH.exists():
        raise FileNotFoundError(
            f"Q-table not found: {QTABLE_PATH}. Run scripts/train_models.py."
        )
    assert QTABLE_PATH.stem.endswith(f"seed{RANDOM_SEED}"), \
        f"Artefact seed mismatch: {QTABLE_PATH.stem}"
    return np.load(QTABLE_PATH)


def _build_shap_explainer(
    travel_matrix: np.ndarray,
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
):
    from src.config import ALPHA, BETA, GAMMA, DELTA

    # Build feature vectors for all 190 facility×CDST pairs
    rows = []
    for i, (_, fac_row) in enumerate(facilities_df.iterrows()):
        for j, (_, cdst_row) in enumerate(cdst_df.iterrows()):
            norm_travel = min(float(travel_matrix[i, j]) / 600.0, 1.0)
            norm_queue = float(cdst_row["current_load"]) / max(float(cdst_row["monthly_capacity"]), 1.0)
            cartridge = float(cdst_row["current_load"])  # CDST labs have no cartridge column
            norm_cart_unavail = 1.0 - min(cartridge / 150.0, 1.0)
            rows.append([norm_travel, norm_queue, norm_cart_unavail, 0.0])

    all_bg = np.array(rows, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(all_bg), size=min(50, len(all_bg)), replace=False)
    background = all_bg[idx]

    def weight_fn(X: np.ndarray) -> np.ndarray:
        return ALPHA * X[:, 0] + BETA * X[:, 1] + GAMMA * X[:, 2] + DELTA * X[:, 3]

    explainer = shap.PermutationExplainer(weight_fn, background)
    logger.info("SHAP PermutationExplainer built on %d background samples", len(background))
    return explainer
