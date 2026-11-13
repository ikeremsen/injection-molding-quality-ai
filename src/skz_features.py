"""SKZ feature preparation: corrected settings, feature-set ladder, curve features.

Nothing here looks at labels. Every feature is built per part from that part's
own row only, so building features once for the whole table cannot leak
information between training and test parts.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .loaders import RESULTS_DIR, SKZ_PARQUET, load_skz

CACHE_DIR = RESULTS_DIR / "cache"
DT = 0.005  # DXP sampling interval in seconds (200 Hz, checked in Phase 1)

# ---------------------------------------------------------------------------
# 1) Corrected settings
# ---------------------------------------------------------------------------
# Phase 1 found recording errors in the SET_ columns. The measured actual
# values agree with the experiment tables in data_description.pdf, so we
# rebuild the settings from those tables. Keys: (day, experiment).
SETTING_FIXES = {
    ("2021-12-07", "A16"): {"SET_HoldingPressure1": 60.0},
    ("2021-12-15", "A02"): {"SET_HoldingPressure1": 200.0},
    ("2021-12-15", "A07"): {"SET_InjectionSpeed": 20.0},
    ("2021-12-15", "A08"): {"SET_InjectionSpeed": 20.0},
    # March A32 repeats Dec-7 A01: two-stage holding, 200 bar for 7 s then 30 bar for 18 s
    # (stage order as stored for Dec-7 in SET_HoldingPressure1/Time1 = first stage).
    ("2022-03-17", "A32"): {"SET_HoldingTime1": 7.0, "SET_HoldingTime2": 18.0},
}
# ABS day: SET_TransferVolume is a copy of the cooling time. Values from the PDF table.
ABS_TRANSFER_VOLUME = {
    "A01": 25.50, "A02": 25.50, "A03": 22.75, "A04": 22.75, "A05": 25.00, "A06": 25.00,
    "A07": 20.50, "A08": 20.50, "A09": 24.50, "A10": 24.50, "A11": 21.50, "A12": 21.50,
    "A13": 24.00, "A14": 24.00, "A15": 20.25, "A16": 20.25, "Z01": 24.00,
}
SETTING_COLS = ["SET_CylinderTemperature", "SET_ToolTemperature", "SET_HoldingPressure1",
                "SET_HoldingPressure2", "SET_InjectionSpeed", "SET_BackPressure",
                "SET_HoldingTime1", "SET_HoldingTime2", "SET_CoolingTime", "SET_TransferVolume"]


def corrected_settings(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the SET_ columns with the known recording errors fixed."""
    s = df[SETTING_COLS].copy()
    for (day, exp), fixes in SETTING_FIXES.items():
        m = (df["day"] == day) & (df["MET_ExperimentNumber"] == exp)
        for col, val in fixes.items():
            s.loc[m, col] = val
    m = df["day"] == "2021-12-15"
    s.loc[m, "SET_TransferVolume"] = df.loc[m, "MET_ExperimentNumber"].map(ABS_TRANSFER_VOLUME)
    return s


# ---------------------------------------------------------------------------
# 2) Curve features from the DXP time series
# ---------------------------------------------------------------------------
# Phases are taken from the machine's own trigger signals (1 while the phase runs).
PHASES = {"inj": "DXP_TrigInj1", "hld": "DXP_TrigHld1", "plst": "DXP_TrigPlst1"}
# Signals: what each one means physically
CURVES = {
    "DXP_Inj1PrsAct": "melt pressure (hydraulic, screw side)",
    "DXP_MldCavPrs1Act": "cavity pressure (sensor in the mold)",
    "DXP_Inj1PosAct": "screw position",
    "DXP_Inj1VelAct": "screw speed",
    "DXP_Inj1RpmAct": "screw rotation (plastification)",
    "DXP_DriveSiePlst1TrqAct": "plastification drive torque",
    "DXP_ClpFceAct": "clamping force",
}


def _phase_stats(x: np.ndarray) -> dict:
    """Simple shape descriptors of one signal inside one phase."""
    if len(x) < 2:
        return dict(max=np.nan, mean=np.nan, area=np.nan, slope=np.nan, end=np.nan)
    t = np.arange(len(x)) * DT
    slope = np.polyfit(t, x, 1)[0]
    return dict(max=x.max(), mean=x.mean(), area=x.sum() * DT, slope=slope, end=x[-1])


def _curve_features_one(row) -> dict:
    out = {}
    masks = {ph: np.asarray(row[trig]) > 0.5 for ph, trig in PHASES.items()}
    for ph, m in masks.items():
        out[f"CRV_{ph}_duration"] = m.sum() * DT
    for sig in CURVES:
        x = np.asarray(row[sig], dtype=float)
        short = sig.replace("DXP_", "").replace("Act", "")
        for ph, m in masks.items():
            for k, v in _phase_stats(x[m]).items():
                out[f"CRV_{short}_{ph}_{k}"] = v
    # A few physically targeted features
    cav = np.asarray(row["DXP_MldCavPrs1Act"], dtype=float)
    inj = masks["inj"]
    i0 = np.flatnonzero(inj)[0] if inj.any() else 0
    out["CRV_cav_time_to_peak"] = (np.argmax(cav) - i0) * DT           # how fast the cavity pressurises
    out["CRV_cav_at_switchover"] = cav[np.flatnonzero(inj)[-1]] if inj.any() else np.nan
    hld = masks["hld"]
    if hld.any():
        h = cav[hld]
        out["CRV_cav_hold_decay"] = h[0] - h[-1]                        # pressure lost during holding (gate freeze)
        out["CRV_cav_hold_time_above_10bar"] = (h > 10).sum() * DT     # how long packing pressure reaches the cavity
    pos = np.asarray(row["DXP_Inj1PosAct"], dtype=float)
    out["CRV_screw_min_pos_cycle"] = pos.min()                           # ~cushion
    return out


def curve_features(force: bool = False) -> pd.DataFrame:
    """Per-part curve features, cached to results/cache (index = MET_MachineCycleID)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / "skz_curve_features.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    cols = ["MET_MachineCycleID"] + list(PHASES.values()) + list(CURVES)
    raw = pq.read_table(SKZ_PARQUET, columns=cols).to_pandas()
    feats = pd.DataFrame([_curve_features_one(r) for _, r in raw.iterrows()],
                         index=raw["MET_MachineCycleID"].values)
    feats.index.name = "MET_MachineCycleID"
    feats.to_parquet(path)
    return feats


# ---------------------------------------------------------------------------
# 3) Feature-set ladder
# ---------------------------------------------------------------------------
# Day/time proxies found in Phase 1. Kept only in the "L3_tc_env" step.
DAY_PROXY_COLS = ["ENV_AirTemperature", "ENV_AirHumidity", "QUA_OilTemperature",
                  "TCE_ActualProcessPowerMean", "TCE_ActualPumpPowerMean",
                  "TCE_ActualPumpSpeedRPMMean", "TCE_ActualPumpSpeedMean",
                  "TCN_ActualPumpSpeedRPMMean", "TCN_ActualPumpSpeedMean"]


def load_skz_model_table():
    """Returns (data, feature_sets).

    data: one row per part with the corrected settings, scalar machine values,
          curve features, labels, targets and IDs (day, group).
    feature_sets: dict name -> list of input columns.
    """
    df = load_skz(with_arrays=False)
    df["day"] = pd.to_datetime(df["MET_TimeCycleStart"]).dt.strftime("%Y-%m-%d")
    df["group"] = df["day"] + "_" + df["MET_ExperimentNumber"]
    df[SETTING_COLS] = corrected_settings(df)

    crv = curve_features()
    df = df.join(crv, on="MET_MachineCycleID")

    def numeric(cols):
        return [c for c in cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True) > 1]

    qua = numeric([c for c in df.columns if c.startswith(("QUA_", "CALC_"))]
                  + [c for c in df.columns if c.startswith("DXP_")])  # DXP scalars only (arrays dropped)
    tc = numeric([c for c in df.columns if c.startswith(("TCE_", "TCN_")) and c.endswith("Mean")])
    env = numeric([c for c in df.columns if c.startswith("ENV_")])
    crv_cols = numeric(list(crv.columns))
    settings = numeric(SETTING_COLS)

    cycle = [c for c in qua if c not in DAY_PROXY_COLS]  # oil temperature = warm-up clock
    cycle_raw = qua
    clean = lambda cols: [c for c in cols if c not in DAY_PROXY_COLS]  # noqa: E731
    feature_sets = {
        "L1_settings": settings,
        "L2_cycle": cycle,
        "L3_tc_env": cycle_raw + tc + env,                          # as in the brief (contains day proxies)
        "L3_clean": clean(cycle + tc),                           # same, day proxies removed
        "L4_curves": clean(cycle + tc) + crv_cols,               # + curve features
    }
    return df, feature_sets
