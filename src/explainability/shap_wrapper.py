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
    travel_min = edge["travel_min"]
    queue_pct = edge["norm_queue"] * 100
    cart_pct = (1 - edge["norm_cart_unavail"]) * 100

    # Sort features: most favorable (negative SHAP) first
    ranked = sorted(zip(feature_names, shap_values), key=lambda t: t[1])

    drivers, limiters = [], []
    for name, val in ranked:
        if abs(val) < 0.005:
            continue
        if name == "travel_time":
            if val < 0:
                drivers.append(f"short travel distance ({travel_min:.0f} min)")
            else:
                limiters.append(f"long travel distance ({travel_min:.0f} min)")
        elif name == "queue_load":
            if val < 0:
                drivers.append(f"low queue pressure ({queue_pct:.0f}% capacity used)")
            else:
                limiters.append(f"high queue pressure ({queue_pct:.0f}% capacity used)")
        elif name == "cartridge_unavail":
            if val < 0:
                drivers.append(f"strong cartridge stock ({cart_pct:.0f}% available)")
            else:
                limiters.append(f"low cartridge stock ({cart_pct:.0f}% available)")
        elif name == "anomaly_score" and val > 0.005:
            limiters.append("anomaly flag active at origin facility")

    parts = []
    if drivers:
        parts.append("Recommended because of " + " and ".join(drivers[:2]))
    if limiters:
        parts.append("limiting factor: " + "; ".join(limiters[:2]))

    disclaimer = (
        " [AI-generated explanation — interpret with caution and validate "
        "against real-world conditions before acting.]"
    )

    if parts:
        return ". ".join(parts) + "." + disclaimer
    return (
        f"{cdst_name} has the lowest composite routing cost across all 5 CDST labs."
        + disclaimer
    )
