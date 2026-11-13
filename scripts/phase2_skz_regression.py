"""Phase 2b: SKZ part-weight regression (SCA_PartWeight). Feature ladder x models x validation.

Run from `05 Code`:
    python scripts/phase2_skz_regression.py
Writes results/tables/phase2_reg_runs.csv and phase2_reg_summary.csv.

Why R^2 "within day": ABS parts weigh ~111 g and PP parts ~91-96 g. A model that
only recognises the material already explains most of the total variance, so the
overall R^2 looks great without saying anything about process control. R^2
computed inside each day measures what matters: telling good and bad settings
apart for the same material.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR  # noqa: E402
from src.modeling import (REGRESSORS, SEEDS, cross_day_splits, drop_unavailable,  # noqa: E402
                          make_regressor, regression_metrics)
from src.skz_features import load_skz_model_table  # noqa: E402

TARGET = "SCA_PartWeight"
CROSS_DAY_SEEDS = SEEDS[:3]
OUT = RESULTS_DIR / "tables"

df, FEATURE_SETS = load_skz_model_table()
GROUPS, DAYS = df["group"].values, df["day"].values
Y = df[TARGET].values
MATERIAL = df["MET_MaterialName"].values  # used ONLY by the reference baseline below, never as a model input


def material_mean(tr, te):
    """Reference baseline: training mean weight of the same material."""
    means = pd.Series(Y[tr]).groupby(MATERIAL[tr]).mean()
    return pd.Series(MATERIAL[te]).map(means).fillna(Y[tr].mean()).values


def predict(model_name, seed, cols, tr, te):
    if model_name == "material_mean":
        return material_mean(tr, te)
    m = make_regressor(model_name, seed).fit(df[cols].iloc[tr], Y[tr])
    return m.predict(df[cols].iloc[te])


def run_group_cv(fs, model_name, seed):
    cols = FEATURE_SETS[fs]
    pred = np.zeros(len(Y))
    for tr, te in GroupKFold(5, shuffle=True, random_state=seed).split(df, Y, GROUPS):
        pred[te] = predict(model_name, seed, cols, tr, te)
    row = regression_metrics(Y, pred, DAYS)
    row.update(scheme="group_5fold", test="all (pooled out-of-fold)", features=fs,
               model=model_name, seed=seed, n_features=len(cols))
    return [row]


def pp_only_splits():
    """Train on one PP day, test on the other PP day (the material is always known on a real machine)."""
    pp_days = ["2021-12-07", "2022-03-17"]
    for test in pp_days:
        train = [d for d in pp_days if d != test][0]
        yield "PP_day_to_PP_day", test, np.flatnonzero(DAYS == train), np.flatnonzero(DAYS == test)


def run_cross_day(fs, model_name, seed):
    rows = []
    for scheme, test_name, tr, te in list(cross_day_splits(df)) + list(pp_only_splits()):
        cols = drop_unavailable(FEATURE_SETS[fs], df.iloc[tr], df.iloc[te])
        pred = predict(model_name, seed, cols, tr, te)
        row = regression_metrics(Y[te], pred, DAYS[te])
        # Pearson r inside the test day: does the model at least RANK parts correctly,
        # even if its absolute level (offset) is wrong?
        row["r_within_test"] = np.corrcoef(Y[te], pred)[0, 1] if np.std(pred) > 0 else np.nan
        row.update(scheme=scheme, test=test_name, features=fs, model=model_name, seed=seed,
                   n_features=len(cols))
        rows.append(row)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    models = ["material_mean"] + REGRESSORS
    tasks = []
    for fs in FEATURE_SETS:
        for m in models:
            if m == "material_mean" and fs != "L1_settings":
                continue  # does not use features; run once
            tasks += [delayed(run_group_cv)(fs, m, s) for s in SEEDS]
            tasks += [delayed(run_cross_day)(fs, m, s) for s in CROSS_DAY_SEEDS]
    results = Parallel(n_jobs=-1, verbose=2)(tasks)
    runs = pd.DataFrame([r for rows in results for r in rows])
    runs.loc[runs.model == "material_mean", "features"] = "(material only)"
    runs.to_csv(OUT / "phase2_reg_runs.csv", index=False)
    metric_cols = [c for c in runs.columns if c.startswith(("mae", "r2", "r_within"))]
    summ = (runs.groupby(["scheme", "test", "features", "model"], sort=False)[metric_cols]
            .agg(["mean", "std"]))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / "phase2_reg_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
