import networkx as nx
import numpy as np
import pandas as pd

from src.config import ALPHA, BETA, GAMMA, DELTA


def build_graph(
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
    travel_matrix: np.ndarray,
    anomaly_scores: pd.Series | None = None,
    overrides: dict[str, int] | None = None,
    delta: float = DELTA,
) -> nx.DiGraph:
    """Build immutable weighted DiGraph. Never mutate after return."""
    overrides = overrides or {}
    G = nx.DiGraph()

    for _, row in facilities_df.iterrows():
        G.add_node(row["facility_id"], **row.to_dict(), node_type="facility")

    for _, row in cdst_df.iterrows():
        G.add_node(row["cdst_id"], **row.to_dict(), node_type="cdst")

    for i, fac_row in facilities_df.iterrows():
        fac_id = fac_row["facility_id"]
        anom = float(anomaly_scores[fac_id]) if anomaly_scores is not None else 0.0

        for j, cdst_row in cdst_df.iterrows():
            cdst_id = cdst_row["cdst_id"]
            cartridge = overrides.get(cdst_id, int(cdst_row["cartridge_availability"])
                                      if "cartridge_availability" in cdst_row
                                      else int(cdst_row["current_load"]))

            travel_min = float(travel_matrix[i, j])
            queue_load = float(cdst_row["current_load"])
            max_cap = float(cdst_row["monthly_capacity"])
            norm_travel = min(travel_min / 600.0, 1.0)
            norm_queue = queue_load / max(max_cap, 1.0)
            norm_cart_unavail = 1.0 - min(float(cartridge) / 150.0, 1.0)

            weight = (ALPHA * norm_travel
                      + BETA * norm_queue
                      + GAMMA * norm_cart_unavail
                      + delta * anom)
            G.add_edge(
                fac_id, cdst_id,
                weight=weight,
                norm_travel=norm_travel,
                norm_queue=norm_queue,
                norm_cart_unavail=norm_cart_unavail,
                anomaly=anom,
                travel_min=travel_min,
            )

    return G


