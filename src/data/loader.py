import pandas as pd
from src.config import FACILITIES_CSV, CDST_CSV


def load_facilities() -> pd.DataFrame:
    return pd.read_csv(FACILITIES_CSV)


def load_cdst_labs() -> pd.DataFrame:
    return pd.read_csv(CDST_CSV)
