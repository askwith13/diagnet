import networkx as nx

from src.config import CDST_IDS


def run_dijkstra(graph: nx.DiGraph, origin_id: str) -> dict:
    """Return {"cdst_id": str, "score": float, "path": list[str]}."""
    best = None
    for cdst_id in CDST_IDS:
        if not graph.has_node(cdst_id):
            continue
        try:
            path = nx.dijkstra_path(graph, origin_id, cdst_id, weight="weight")
            score = nx.dijkstra_path_length(graph, origin_id, cdst_id, weight="weight")
        except nx.NetworkXNoPath:
            continue
        if best is None or score < best["score"]:
            best = {"cdst_id": cdst_id, "score": score, "path": path}

    if best is None:
        raise ValueError(f"No path found from {origin_id} to any CDST lab")
    return best
