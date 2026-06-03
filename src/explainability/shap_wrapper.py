import numpy as np
import networkx as nx

from src.config import CDST_NAMES


def explain_routing(
    origin_id: str,
    recommended_cdst_id: str,
    graph: nx.DiGraph,
    explainer,
) -> dict:
    """Return {"shap_values": np.ndarray, "feature_names": list[str], "explanation_text": str}."""
    feature_names = ["travel_time", "queue_load", "cartridge_unavail", "anomaly_score"]

    edge = graph.edges[origin_id, recommended_cdst_id]
    instance = np.array([[
        edge["norm_travel"],
        edge["norm_queue"],
        edge["norm_cart_unavail"],
        edge["anomaly"],
    ]])

    sv = explainer(instance)
    shap_values = sv.values[0]  # shape (4,)

    explanation_text = _generate_explanation(
        origin_id, recommended_cdst_id, shap_values, feature_names, edge
    )
    return {
        "shap_values": shap_values,
        "feature_names": feature_names,
        "explanation_text": explanation_text,
    }


def _generate_explanation(
    origin_id: str,
    cdst_id: str,
    shap_values: np.ndarray,
    feature_names: list[str],
    edge: dict,
) -> str:
    cdst_name = CDST_NAMES.get(cdst_id, cdst_id)

    # SHAP values explain the routing cost (lower = better).
    # Negative SHAP → feature reduces cost → favorable for this recommendation.
    labels = {
        "travel_time": f"travel time ({edge['travel_min']:.0f} min)",
        "queue_load": "lab queue load",
        "cartridge_unavail": "cartridge availability",
        "anomaly_score": "anomaly flag",
    }

    # Sort by contribution magnitude; negative = favorable
    ranked = sorted(zip(feature_names, shap_values), key=lambda t: t[1])

    parts = []
    for name, val in ranked:
        if abs(val) < 0.005:
            continue
        label = labels[name]
        if val < 0:
            parts.append(f"favorable {label}")
        else:
            parts.append(f"elevated {label}")

    if parts:
        factors = "; ".join(parts[:3])
        return f"{cdst_name} recommended — key factors: {factors}."
    return f"{cdst_name} recommended as the lowest-cost referral destination."
