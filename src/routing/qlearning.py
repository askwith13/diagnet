import numpy as np
import pandas as pd

from src.config import CDST_IDS, N_CDST


def run_qlearning(
    qtable: np.ndarray,
    origin_id: str,
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
) -> dict:
    """Return {"cdst_id": str, "q_value": float}. Inference only — no training."""
    state_idx = _encode_state(origin_id, facilities_df)
    action_idx = int(np.argmax(qtable[state_idx]))
    return {
        "cdst_id": CDST_IDS[action_idx],
        "q_value": float(qtable[state_idx, action_idx]),
    }


def _encode_state(origin_id: str, facilities_df: pd.DataFrame) -> int:
    # State = facility row index (0–37). Q-table shape: (N_FACILITIES, N_ACTIONS).
    # Dynamic CDST conditions are averaged out during training via episode sampling.
    idx = facilities_df.index[facilities_df["facility_id"] == origin_id].tolist()
    if not idx:
        raise ValueError(f"facility_id {origin_id!r} not found in facilities data")
    return idx[0]
