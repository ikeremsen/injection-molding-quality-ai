"""Loaders for the three public datasets.

Every function only READS the raw files. Nothing here writes back to the raw
data folders. All paths are relative to the data root (PROJECT_ROOT below).

Usage (from inside the code folder):
    from src.loaders import load_skz, load_leon, load_polenta
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
import os

CODE_DIR = Path(__file__).resolve().parents[1]          # the code folder (this repository)
# Folder that holds the raw datasets (see README, "Getting the data").
# Default: the parent of the code folder. Override with the IMAI_DATA_ROOT environment variable.
PROJECT_ROOT = Path(os.environ.get("IMAI_DATA_ROOT", CODE_DIR.parent))
RESULTS_DIR = CODE_DIR / "results"

SKZ_DIR = PROJECT_ROOT / "ZIPS" / "k0v7s-jf859"
SKZ_PARQUET = SKZ_DIR / "dataset_V2.parquet"
SKZ_PARAM_NAMES = SKZ_DIR / "DataXplorer-Parameter_Names.xlsx"

LEON_CSV = PROJECT_ROOT / "02 Datasets" / "LeonPP.csv"

_POLENTA_REPO = (PROJECT_ROOT / "ZIPS"
                 / "machine-learning-for-quality-prediction-in-plastic-injection-molding-master"
                 / "machine-learning-for-quality-prediction-in-plastic-injection-molding-master")
POLENTA_CSV = _POLENTA_REPO / "dataset" / "data.csv"


# ---------------------------------------------------------------------------
# A) SKZ / Fraunhofer IPA
# ---------------------------------------------------------------------------
# Column prefixes we may use as model INPUTS, and prefixes that must never be
# inputs (labels, targets, metadata, or information measured after the part
# left the machine).
SKZ_INPUT_PREFIXES = ("SET_", "QUA_", "DXP_", "CALC_", "TCE_", "TCN_", "ENV_")
SKZ_FORBIDDEN_PREFIXES = ("LBL_", "SCA_", "CV_", "IR_", "SIM_", "MET_")


def _is_array_cell(x) -> bool:
    return isinstance(x, (list, np.ndarray))


def skz_array_columns(df: pd.DataFrame) -> list[str]:
    """Columns whose cells hold a time series (one array per part)."""
    out = []
    for c in df.columns:
        first = df[c].dropna()
        if len(first) and _is_array_cell(first.iloc[0]):
            out.append(c)
    return out


def load_skz(with_arrays: bool = True) -> pd.DataFrame:
    """Load the SKZ parquet. One row = one molded part.

    with_arrays=False drops the time-series columns, which makes the frame
    small and fast for scalar-only work.
    """
    df = pd.read_parquet(SKZ_PARQUET)
    if not with_arrays:
        df = df.drop(columns=skz_array_columns(df))
    return df


def skz_group_id(df: pd.DataFrame, date_col: str, exp_col: str = "MET_ExperimentNumber") -> pd.Series:
    """Group ID = test date + experiment number.

    Experiment numbers restart every test day, so the number alone would put
    parts from different days (and materials) in the same group.
    """
    date = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")
    return date + "_E" + df[exp_col].astype(int).astype(str).str.zfill(2)


# ---------------------------------------------------------------------------
# B) León PP
# ---------------------------------------------------------------------------
def load_leon() -> pd.DataFrame:
    """1000 batches (rows) x 5 parts. Comma-separated, empty cells = NaN."""
    return pd.read_csv(LEON_CSV)


# ---------------------------------------------------------------------------
# C) Polenta et al.
# ---------------------------------------------------------------------------
def load_polenta() -> pd.DataFrame:
    """1451 lenses. SEMICOLON-separated. Row order is kept (assumed chronological)."""
    df = pd.read_csv(POLENTA_CSV, sep=";")
    df["quality"] = df["quality"].astype(int)
    return df
