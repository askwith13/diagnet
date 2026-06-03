import logging
import os

import numpy as np
import pandas as pd
import streamlit as st

from src.config import ORS_MATRIX_CACHE, ORS_BASE_URL

logger = logging.getLogger(__name__)


@st.cache_data
def load_or_fetch_ors_matrix(
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
) -> tuple[np.ndarray, bool]:
    """Return (38×5 travel-time matrix in minutes, is_fallback)."""
    if ORS_MATRIX_CACHE.exists():
        logger.info("Loading ORS matrix from cache")
        return np.load(ORS_MATRIX_CACHE), False

    try:
        matrix = _call_ors_api(facilities_df, cdst_df)
        np.save(ORS_MATRIX_CACHE, matrix)
        return matrix, False
    except Exception as exc:
        logger.warning("ORS API unavailable (%s) — using Haversine fallback", exc)
        return _haversine_matrix(facilities_df, cdst_df), True


def _call_ors_api(facilities_df: pd.DataFrame, cdst_df: pd.DataFrame) -> np.ndarray:
    """Call ORS Matrix API. Returns (N_FAC × N_CDST) travel-time matrix in minutes."""
    api_key = os.getenv("ORS_API_KEY")
    if not api_key:
        raise ValueError("ORS_API_KEY not set — check .env file")

    import openrouteservice
    client = openrouteservice.Client(key=api_key)

    # ORS expects [lon, lat] — note reversed order from our CSV (lat, lon)
    fac_coords = facilities_df[["lon", "lat"]].values.tolist()
    cdst_coords = cdst_df[["lon", "lat"]].values.tolist()
    all_coords = fac_coords + cdst_coords

    n_fac = len(fac_coords)
    sources = list(range(n_fac))
    destinations = list(range(n_fac, n_fac + len(cdst_coords)))

    logger.info("Calling ORS Matrix API: %d sources × %d destinations", len(sources), len(destinations))
    response = client.distance_matrix(
        locations=all_coords,
        sources=sources,
        destinations=destinations,
        metrics=["duration"],
        profile="driving-car",
    )

    # response["durations"] is list[list[float|None]] in seconds
    raw = response["durations"]
    matrix = np.array(
        [[d / 60.0 if d is not None else 999.0 for d in row] for row in raw],
        dtype=float,
    )
    logger.info("ORS matrix received: min=%.1f min, max=%.1f min", matrix.min(), matrix.max())
    return matrix


def _haversine_matrix(
    facilities_df: pd.DataFrame,
    cdst_df: pd.DataFrame,
    speed_kmh: float = 45.0,
) -> np.ndarray:
    """Straight-line distance ÷ assumed road speed. Underestimates real travel times."""
    R = 6371.0
    fac_lats = np.radians(facilities_df["lat"].values)
    fac_lons = np.radians(facilities_df["lon"].values)
    cdst_lats = np.radians(cdst_df["lat"].values)
    cdst_lons = np.radians(cdst_df["lon"].values)

    matrix = np.zeros((len(facilities_df), len(cdst_df)))
    for j in range(len(cdst_df)):
        dlat = cdst_lats[j] - fac_lats
        dlon = cdst_lons[j] - fac_lons
        a = (np.sin(dlat / 2) ** 2
             + np.cos(fac_lats) * np.cos(cdst_lats[j]) * np.sin(dlon / 2) ** 2)
        km = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        matrix[:, j] = km / speed_kmh * 60.0

    logger.info("Haversine matrix: min=%.1f min, max=%.1f min", matrix.min(), matrix.max())
    return matrix
