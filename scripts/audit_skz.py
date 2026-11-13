"""Phase 1 audit for the SKZ / Fraunhofer IPA dataset. No modeling.

Run from the `05 Code` folder:
    python scripts/audit_skz.py
Writes:
    results/audit/audit_skz.txt            human-readable report
    results/audit/skz_columns.csv          one row per column (prefix, kind, missing, constant...)
    results/audit/skz_labels_by_group.csv  defect rates per day x experiment
    results/audit/skz_settings_by_group.csv SET_ values per group (check vs. PDF tables)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import (RESULTS_DIR, SKZ_FORBIDDEN_PREFIXES,  # noqa: E402
                         SKZ_INPUT_PREFIXES, load_skz, skz_array_columns)

OUT = RESULTS_DIR / "audit"
OUT.mkdir(parents=True, exist_ok=True)
lines: list[str] = []


def say(s=""):
    lines.append(str(s))


df = load_skz()
arr_cols = skz_array_columns(df)
df["day"] = pd.to_datetime(df["MET_TimeCycleStart"]).dt.strftime("%Y-%m-%d")
df["group"] = df["day"] + "_" + df["MET_ExperimentNumber"]
DAYS = sorted(df["day"].unique())

# ---------------------------------------------------------------------------
say("=== SKZ: basic shape ===")
say(f"rows: {len(df)}, original columns: {df.shape[1] - 2}, array columns: {len(arr_cols)}")
say(f"MET_MachineCycleID unique: {df['MET_MachineCycleID'].is_unique}")
say("\nrows / experiments / material per day:")
say(df.groupby("day").agg(rows=("group", "size"), groups=("group", "nunique"),
                          material=("MET_MaterialName", lambda s: ",".join(s.unique())),
                          moisture=("MET_MaterialMoisture", lambda s: ",".join(map(str, s.unique())))).to_string())
say(f"total groups (day+experiment): {df['group'].nunique()}")
say(f"parts per group: {df.groupby('group').size().value_counts().to_dict()}")
say("experiment numbers per day:")
for d in DAYS:
    say(f"  {d}: {sorted(df.loc[df.day == d, 'MET_ExperimentNumber'].unique())}")
# Are experiments run in contiguous blocks (12 cycles in a row)?
blocks = (df.sort_values("MET_TimeCycleStart")["group"]
          .pipe(lambda s: (s != s.shift()).sum()))
say(f"number of contiguous runs of a group in time order: {blocks} (47 = each group run once, in one block)")

# ---------------------------------------------------------------------------
say("\n=== Column inventory per prefix ===")
rows = []
for c in df.columns.drop(["day", "group"]):
    pre = c.split("_")[0]
    is_arr = c in arr_cols
    s = df[c]
    if is_arr:
        lens = s.map(lambda a: len(a) if a is not None else 0)
        n_missing = int((lens == 0).sum())
        # constant = every array has the same min and max (i.e. flat, identical values)
        mins = s.map(lambda a: np.nanmin(a) if a is not None and len(a) else np.nan)
        maxs = s.map(lambda a: np.nanmax(a) if a is not None and len(a) else np.nan)
        constant = bool(np.nanmax(maxs) == np.nanmin(mins)) if mins.notna().any() else True
        n_unique = np.nan
        extra = f"len {lens.min()}-{lens.max()}"
    else:
        n_missing = int(s.isna().sum())
        n_unique = s.nunique(dropna=True)
        constant = n_unique <= 1
        extra = ""
    role = ("INPUT-allowed" if c.startswith(SKZ_INPUT_PREFIXES)
            else "FORBIDDEN" if c.startswith(SKZ_FORBIDDEN_PREFIXES) else "UNDECIDED")
    rows.append(dict(column=c, prefix=pre, kind="array" if is_arr else "scalar",
                     role=role, n_missing=n_missing, n_unique=n_unique,
                     constant=constant, note=extra))
inv = pd.DataFrame(rows)
# Missing by day for scalars and arrays
for d in DAYS:
    m = df["day"] == d
    inv[f"missing_{d}"] = [
        int(df.loc[m, c].map(lambda a: a is None or len(a) == 0).sum()) if c in arr_cols
        else int(df.loc[m, c].isna().sum())
        for c in inv["column"]]
inv.to_csv(OUT / "skz_columns.csv", index=False)

say(inv.groupby(["prefix", "kind"]).agg(
    n=("column", "size"), role=("role", "first"),
    any_missing=("n_missing", lambda x: int((x > 0).sum())),
    constant=("constant", "sum")).to_string())
say("\nUNDECIDED prefixes (not in the allowed or forbidden list): "
    + str(sorted(inv.loc[inv.role == "UNDECIDED", "prefix"].unique())))

say("\nconstant columns (no information at all):")
say(", ".join(inv.loc[inv.constant, "column"]))

say("\nscalar columns with missing values (count per day):")
miss = inv[(inv.kind == "scalar") & (inv.n_missing > 0)]
say(miss[["column", "role"] + [f"missing_{d}" for d in DAYS]].to_string(index=False))
say("\narray columns with missing arrays (count per day):")
miss = inv[(inv.kind == "array") & (inv.n_missing > 0)]
say(miss[["column", "role"] + [f"missing_{d}" for d in DAYS]].to_string(index=False))

# Columns that are constant INSIDE every day (only useful for telling days apart)
say("\nallowed scalar inputs that only vary BETWEEN days (constant within each day):")
allowed_sc = inv[(inv.role == "INPUT-allowed") & (inv.kind == "scalar") & ~inv.constant]["column"]
day_only = [c for c in allowed_sc if (df.groupby("day")[c].nunique(dropna=True) <= 1).all()]
say(", ".join(day_only) if day_only else "(none)")

# ---------------------------------------------------------------------------
say("\n=== Array sampling check ===")
for c in ["DXP_Inj1PrsAct", "TCE_TemperatureMainLine", "DOS_acWeight", "DRY_acTempDryair" if "DRY_acTempDryair" in df else "DRY_HT101_acTempDryair"]:
    lens = df[c].map(lambda a: len(a) if a is not None else 0)
    ratio = lens / df["QUA_CycleTime"]
    say(f"{c}: length / cycle time = {ratio.median():.1f} samples per s "
        f"(min {ratio.min():.1f}, max {ratio.max():.1f})")

# ---------------------------------------------------------------------------
say("\n=== Labels ===")
lbl = [c for c in df.columns if c.startswith("LBL_")]
say("overall positive rate:\n" + df[lbl].mean().round(3).to_string())
say("\nNOK rate per day and count:")
say(df.groupby("day")["LBL_NOK"].agg(["sum", "size", "mean"]).round(3).to_string())
say("\nall labels by day (rate):")
say(df.groupby("day")[lbl].mean().round(3).T.to_string())
nok_rule = (df["LBL_SinkMarks"].astype(bool) | df["LBL_Underfilled"].astype(bool)).astype(int)
say(f"\nrows where LBL_NOK != (SinkMarks OR Underfilled): {(nok_rule != df['LBL_NOK']).sum()}")
say(f"rows with BOTH sink marks and underfilled: {int((df.LBL_SinkMarks.astype(bool) & df.LBL_Underfilled.astype(bool)).sum())}")
grp = df.groupby(["day", "MET_ExperimentNumber"])[lbl].mean().round(2)
grp.to_csv(OUT / "skz_labels_by_group.csv")
nok_g = df.groupby("group")["LBL_NOK"].mean()
say("\nNOK rate per group: how 'pure' are groups?")
say(f"  groups with NOK rate 0: {(nok_g == 0).sum()}, rate 1: {(nok_g == 1).sum()}, "
    f"mixed (0<rate<1): {((nok_g > 0) & (nok_g < 1)).sum()} of {len(nok_g)}")
say("  mixed groups: " + ", ".join(f"{g}={v:.2f}" for g, v in nok_g[(nok_g > 0) & (nok_g < 1)].items()))
for c in ["LBL_SinkMarks", "LBL_Underfilled"]:
    r = df.groupby("group")[c].mean()
    say(f"  {c}: groups all-1 {(r == 1).sum()}, all-0 {(r == 0).sum()}, mixed {((r > 0) & (r < 1)).sum()}")
# Within mixed groups: are the NOK parts the first ones (warm-up after a setting change)?
df["pos_in_group"] = df.sort_values("MET_TimeCycleStart").groupby("group").cumcount() + 1
mixed = df[df.group.isin(nok_g[(nok_g > 0) & (nok_g < 1)].index)]
say("  NOK rate by part position (1=first part after setting change), mixed groups only:")
say("  " + mixed.groupby("pos_in_group")["LBL_NOK"].mean().round(2).to_dict().__str__())

# ---------------------------------------------------------------------------
say("\n=== Regression targets ===")
for c in ["SCA_PartWeight", "CV_Warpage"]:
    say(f"{c}:")
    say(df.groupby("day")[c].describe().round(3).to_string())
    # how much of the variance is between groups vs inside groups
    within = df.groupby("group")[c].std().median()
    say(f"  median within-group std: {within:.3f}; overall std: {df[c].std():.3f}")
say("\npart weight by NOK (does weight already 'know' the label?):")
say(df.groupby(["day", "LBL_NOK"])["SCA_PartWeight"].mean().round(2).unstack().to_string())

# ---------------------------------------------------------------------------
say("\n=== Settings per group (compare with PDF tables) ===")
set_cols = [c for c in df.columns if c.startswith("SET_")]
sets = df.groupby(["day", "MET_ExperimentNumber"])[set_cols].agg(lambda s: s.unique()[0] if s.nunique(dropna=False) == 1 else "VARIES")
sets.to_csv(OUT / "skz_settings_by_group.csv")
say(sets.to_string())
n_unique_settings = df[set_cols].fillna(-1).drop_duplicates().shape[0]
say(f"\ndistinct SET_ combinations across all 47 groups: {n_unique_settings}")

# Setpoint vs. actual (from the PDF mapping table)
say("\nsetpoint vs actual value (median |actual - set| per day):")
pairs = [("SET_CylinderTemperature", "QUA_CylinderTemperature11"),
         ("SET_ToolTemperature", "TCE_TemperatureMainLineMean"),
         ("SET_InjectionSpeed", "QUA_InjectionSpeed1"),
         ("SET_HoldingPressure1", "DXP_HoldingPressure1"),
         ("SET_HoldingPressure2", "DXP_HoldingPressure2"),
         ("SET_HoldingTime1", "DXP_HoldingTime1"),
         ("SET_HoldingTime2", "DXP_HoldingTime2"),
         ("SET_CoolingTime", "QUA_CoolingTime"),
         ("SET_TransferVolume", "QUA_TransferStroke")]
for s, a in pairs:
    if s in df and a in df:
        dev = (df[a] - df[s]).abs().groupby(df["day"]).median().round(2).to_dict()
        corr = df[[s, a]].corr().iloc[0, 1]
        say(f"  {s:28s} vs {a:30s} corr {corr:5.2f}  median|diff| {dev}")

# ---------------------------------------------------------------------------
say("\n=== Leakage / proxy checks on ALLOWED inputs ===")
# 1) Monotone counters or totals: they encode run order (= time), which is a
#    proxy for the experiment ID, not physics.
order = df["MET_MachineCycleID"]
sc_allowed = inv[(inv.role == "INPUT-allowed") & (inv.kind == "scalar") & ~inv.constant]["column"].tolist()
cor = {c: df[c].corr(order, method="spearman") for c in sc_allowed}
say("allowed scalars with |Spearman corr| > 0.9 to machine cycle number (time proxies):")
say("  " + str({k: round(v, 3) for k, v in cor.items() if abs(v) > 0.9}))
# Same check within each day
say("allowed scalars with |Spearman corr| > 0.95 to cycle number WITHIN every day:")
within = []
for c in sc_allowed:
    rs = [df.loc[df.day == d, c].corr(df.loc[df.day == d, "MET_MachineCycleID"], method="spearman") for d in DAYS]
    if all(abs(r) > 0.95 for r in rs if pd.notna(r)) and any(pd.notna(r) for r in rs):
        within.append(c)
say("  " + (", ".join(within) if within else "(none)"))
# Array columns that look like counters (last value - first value grows with time)
say("array columns whose name suggests a counter / cumulative total:")
say("  " + ", ".join(c for c in arr_cols if any(k in c for k in ("Counter", "Total", "BatchCounter"))))

# 2) Allowed scalars that almost perfectly separate NOK on their own.
from sklearn.metrics import roc_auc_score  # noqa: E402  (only used for a quick 1-feature check)
auc = {}
for c in sc_allowed:
    x = df[c]
    ok = x.notna()
    if x[ok].nunique() > 1:
        a = roc_auc_score(df.loc[ok, "LBL_NOK"], x[ok])
        auc[c] = max(a, 1 - a)
top = pd.Series(auc).sort_values(ascending=False).head(15)
say("\nsingle allowed scalar vs LBL_NOK, ROC-AUC on ALL rows (no CV; screening only):")
say(top.round(3).to_string())

# ---------------------------------------------------------------------------
say("\n=== Environment ===")
say(df.groupby("day")[["ENV_AirTemperature", "ENV_AirHumidity"]].agg(["min", "max"]).round(1).to_string())
say(f"ENV within-group std (median): temp {df.groupby('group')['ENV_AirTemperature'].std().median():.2f}, "
    f"humidity {df.groupby('group')['ENV_AirHumidity'].std().median():.2f}")

text = "\n".join(lines)
(OUT / "audit_skz.txt").write_text(text, encoding="utf-8")
print(text)
