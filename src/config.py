from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"
CACHE_DIR = ROOT / "cache"

FACILITIES_CSV = DATA_DIR / "facilities.csv"
CDST_CSV = DATA_DIR / "cdst_labs.csv"
MANIFEST_JSON = ARTIFACTS_DIR / "manifest.json"
ORS_MATRIX_CACHE = CACHE_DIR / "ors_matrix.npy"

RANDOM_SEED = 42

ISOLATION_FOREST_PATH = ARTIFACTS_DIR / f"isolation_forest_seed{RANDOM_SEED}.joblib"
QTABLE_PATH = ARTIFACTS_DIR / f"qtable_seed{RANDOM_SEED}.npy"

# ── Routing weight defaults (FR-R1) ──────────────────────────────────────────
ALPHA = 0.5   # travel time weight
BETA = 0.3    # queue load weight
GAMMA = 0.2   # cartridge unavailability weight
DELTA = 0.0   # anomaly score weight (FR-A3, off by default)

# ── ORS ──────────────────────────────────────────────────────────────────────
ORS_BASE_URL = "https://api.openrouteservice.org"

# ── CDST lab IDs (canonical) ─────────────────────────────────────────────────
CDST_IDS = ["CDST-01", "CDST-02", "CDST-03", "CDST-04", "CDST-05"]

CDST_NAMES = {
    "CDST-01": "NTEP State Reference Laboratory, Kolkata",
    "CDST-02": "NBMCH CDST Lab",
    "CDST-03": "Midnapore MCH CDST Lab",
    "CDST-04": "Murshidabad MCH CDST Lab",
    "CDST-05": "Bardhamaan MCH CDST Lab",
}

# ── West Bengal bounding box ──────────────────────────────────────────────────
WB_LAT_MIN, WB_LAT_MAX = 21.5, 27.2
WB_LON_MIN, WB_LON_MAX = 85.8, 89.9

# ── Q-learning ───────────────────────────────────────────────────────────────
N_FACILITIES = 38
N_CDST = 5
N_ACTIONS = N_CDST
