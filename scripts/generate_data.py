"""Run once to generate synthetic West Bengal TB diagnostic network data.

Usage:
    python scripts/generate_data.py
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    RANDOM_SEED, FACILITIES_CSV, CDST_CSV, MANIFEST_JSON,
    WB_LAT_MIN, WB_LAT_MAX, WB_LON_MIN, WB_LON_MAX,
    CDST_IDS, CDST_NAMES, N_FACILITIES,
)

rng = np.random.default_rng(RANDOM_SEED)


def generate_facilities() -> pd.DataFrame:
    """38 district facilities across all 23 WB districts, geographically
    assigned to their nearest CDST lab."""
    # Base records: (facility_name, base_lat, base_lon, assigned_cdst_id, dhc)
    # dhc=True → District Hospital Complex (higher volume); False → CHC
    _BASE = [
        # ── CDST-01: NTEP Kolkata (south / metro belt) ──────────────────────
        ("Kolkata DHC",                    22.5726, 88.3639, "CDST-01", True),
        ("Kolkata NRS CHC",                22.5448, 88.3426, "CDST-01", False),
        ("North 24 Parganas DHC (Barasat)",22.7218, 88.4797, "CDST-01", True),
        ("North 24 Parganas CHC (Basirhat)",22.6573, 88.8797, "CDST-01", False),
        ("South 24 Parganas DHC (Diamond Harbour)", 22.1918, 88.1958, "CDST-01", True),
        ("South 24 Parganas CHC (Canning)",22.3186, 88.6633, "CDST-01", False),
        ("Howrah DHC",                     22.5958, 88.2636, "CDST-01", True),
        ("Hooghly DHC (Chinsurah)",        22.8966, 88.3905, "CDST-01", True),
        ("Nadia DHC (Krishnanagar)",       23.4041, 88.5025, "CDST-01", True),
        ("Nadia CHC (Kalyani)",            22.9750, 88.4344, "CDST-01", False),
        # ── CDST-02: NBMCH Siliguri (north / terai / hills) ──────────────────
        ("Uttar Dinajpur DHC (Raiganj)",   25.6191, 88.1242, "CDST-02", True),
        ("Uttar Dinajpur CHC (Islampur)",  26.2685, 88.1870, "CDST-02", False),
        ("Dakshin Dinajpur DHC (Balurghat)",25.2312, 88.7700, "CDST-02", True),
        ("Jalpaiguri DHC",                 26.5449, 88.7179, "CDST-02", True),
        ("Jalpaiguri CHC (Mal)",           26.8715, 88.7126, "CDST-02", False),
        ("Alipurduar DHC",                 26.4897, 89.5276, "CDST-02", True),
        ("Cooch Behar DHC",                26.3261, 89.4464, "CDST-02", True),
        ("Darjeeling DHC",                 27.0410, 88.2663, "CDST-02", True),
        ("Kalimpong DHC",                  27.0668, 88.4720, "CDST-02", True),
        # ── CDST-03: Midnapore MCH (south-west) ──────────────────────────────
        ("Purba Medinipur DHC (Tamluk)",   22.2832, 87.9226, "CDST-03", True),
        ("Purba Medinipur CHC (Haldia)",   22.0257, 88.0583, "CDST-03", False),
        ("Paschim Medinipur DHC",          22.4239, 87.3191, "CDST-03", True),
        ("Paschim Medinipur CHC (Kharagpur)", 22.3460, 87.3237, "CDST-03", False),
        ("Jhargram DHC",                   22.4588, 86.9924, "CDST-03", True),
        ("Bankura DHC",                    23.2324, 87.0750, "CDST-03", True),
        ("Bankura CHC (Bishnupur)",        23.0770, 87.3150, "CDST-03", False),
        ("Purulia DHC",                    23.3323, 86.3638, "CDST-03", True),
        ("Purulia CHC (Raghunathpur)",     23.5435, 86.6716, "CDST-03", False),
        # ── CDST-04: Murshidabad MCH (central-west / north-central) ──────────
        ("Birbhum DHC (Suri)",             23.9092, 87.5338, "CDST-04", True),
        ("Birbhum CHC (Bolpur)",           23.6699, 87.7152, "CDST-04", False),
        ("Murshidabad DHC (Berhampore)",   24.1058, 88.2447, "CDST-04", True),
        ("Murshidabad CHC (Jangipur)",     24.4682, 88.0738, "CDST-04", False),
        ("Malda DHC (English Bazar)",      25.0029, 88.1431, "CDST-04", True),
        ("Malda CHC (Old Malda)",          25.0268, 88.1371, "CDST-04", False),
        # ── CDST-05: Bardhamaan MCH (central) ────────────────────────────────
        ("Purba Bardhaman DHC (Burdwan)",  23.2324, 87.8615, "CDST-05", True),
        ("Purba Bardhaman CHC (Katwa)",    23.6481, 88.1306, "CDST-05", False),
        ("Paschim Bardhaman DHC (Asansol)",23.6835, 86.9740, "CDST-05", True),
        ("Paschim Bardhaman CHC (Durgapur)",23.5204, 87.3119, "CDST-05", False),
    ]

    records = []
    for i, (name, base_lat, base_lon, cdst_id, is_dhc) in enumerate(_BASE):
        jitter_lat = rng.uniform(-0.04, 0.04)
        jitter_lon = rng.uniform(-0.04, 0.04)

        if is_dhc:
            tests = int(rng.integers(120, 301))
            cartridges = int(rng.integers(40, 151))
        else:
            tests = int(rng.integers(20, 121))
            cartridges = int(rng.integers(0, 81))

        records.append({
            "facility_id": f"FAC-{i+1:02d}",
            "facility_name": name,
            "lat": round(base_lat + jitter_lat, 6),
            "lon": round(base_lon + jitter_lon, 6),
            "tests_per_month": tests,
            "cartridge_availability": cartridges,
            "assigned_cdst_id": cdst_id,
        })

    return pd.DataFrame(records)


def generate_cdst_labs() -> pd.DataFrame:
    # Real CDST labs with approximate coordinates
    records = [
        {"cdst_id": "CDST-01", "cdst_name": CDST_NAMES["CDST-01"], "lat": 22.5726, "lon": 88.3639, "monthly_capacity": 500, "current_load": rng.integers(100, 400)},
        {"cdst_id": "CDST-02", "cdst_name": CDST_NAMES["CDST-02"], "lat": 26.7271, "lon": 88.3953, "monthly_capacity": 300, "current_load": rng.integers(50, 250)},
        {"cdst_id": "CDST-03", "cdst_name": CDST_NAMES["CDST-03"], "lat": 22.4239, "lon": 87.3191, "monthly_capacity": 250, "current_load": rng.integers(50, 200)},
        {"cdst_id": "CDST-04", "cdst_name": CDST_NAMES["CDST-04"], "lat": 24.1800, "lon": 88.2700, "monthly_capacity": 200, "current_load": rng.integers(30, 180)},
        {"cdst_id": "CDST-05", "cdst_name": CDST_NAMES["CDST-05"], "lat": 23.2324, "lon": 87.8615, "monthly_capacity": 280, "current_load": rng.integers(50, 230)},
    ]
    return pd.DataFrame(records)


def write_manifest(facilities_df: pd.DataFrame, cdst_df: pd.DataFrame):
    combined = facilities_df.to_csv(index=False) + cdst_df.to_csv(index=False)
    data_hash = hashlib.sha256(combined.encode()).hexdigest()
    MANIFEST_JSON.parent.mkdir(exist_ok=True)
    MANIFEST_JSON.write_text(json.dumps({
        "random_seed": RANDOM_SEED,
        "data_hash": data_hash,
        "created": pd.Timestamp.now().isoformat(),
    }, indent=2))
    print(f"Manifest written: {MANIFEST_JSON}")


if __name__ == "__main__":
    print(f"Generating data with RANDOM_SEED={RANDOM_SEED}...")
    facilities_df = generate_facilities()
    cdst_df = generate_cdst_labs()
    FACILITIES_CSV.parent.mkdir(exist_ok=True)
    facilities_df.to_csv(FACILITIES_CSV, index=False)
    cdst_df.to_csv(CDST_CSV, index=False)
    write_manifest(facilities_df, cdst_df)
    print(f"Facilities: {FACILITIES_CSV}")
    print(f"CDST labs:  {CDST_CSV}")
