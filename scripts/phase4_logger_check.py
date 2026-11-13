"""Phase 4d: score candidate logger configurations on SKZ (grouped CV).

Run from `05 Code`:  python scripts/phase4_logger_check.py
Writes results/tables/phase4_logger_check.csv

Caveat: the candidate sets were proposed after looking at the Phase 4 selection
results on the same data, so these scores are mildly optimistic. The nested
selection curves (phase4_selection_curves.png) are the unbiased version.
"""
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ns = runpy.run_path(str(Path(__file__).with_name("phase4_skz.py")), run_name="lib")
df, CH_COLS, CHANNELS, TARGETS = ns["df"], ns["CH_COLS"], ns["CHANNELS"], ns["TARGETS"]
score, splits, DAY, X_ALL = ns["score"], ns["splits"], ns["DAY"], ns["X_ALL"]
from src.loaders import RESULTS_DIR  # noqa: E402
from src.modeling import _imp, make_classifier  # noqa: E402

CHEAP = ["screw_position", "melt_pressure", "clamp_force", "mold_temperature", "plastification", "injection_speed"]
CONFIGS = {
    "A. cheap logger, cycle summaries only (no curves)": (CHEAP, False),
    "B. cheap logger with curves": (CHEAP, True),
    "C. B + cavity pressure sensor": (CHEAP + ["cavity_pressure"], True),
    "D. everything measured": ([c for c in CHANNELS if c != "setpoints"], True),
    "E. setpoints only (no logger)": (["setpoints"], False),
}


def cols_for(channels, curves):
    cols = sum((CH_COLS[c] for c in channels), [])
    return cols if curves else [c for c in cols if not c.startswith("CRV_")]


def model(kind, seed):
    if kind == "cls":
        return make_classifier("logreg", seed)
    return make_pipeline(_imp(), RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=1, random_state=seed))


def job(target, cfg, seed):
    kind = TARGETS[target]
    y = df[target].astype(float).values if kind == "reg" else df[target].astype(int).values
    cols = cols_for(*CONFIGS[cfg])
    pred = np.zeros(len(y))
    for tr, te in splits(kind, y, seed):
        m = model(kind, seed).fit(X_ALL.iloc[tr][cols], y[tr])
        pred[te] = m.predict_proba(X_ALL.iloc[te][cols])[:, 1] if kind == "cls" else m.predict(X_ALL.iloc[te][cols])
    s = score(kind, y, pred, DAY)
    return dict(target=target, config=cfg, seed=seed, n_features=len(cols),
                score=-s if kind == "reg" else s, metric="offset-corrected MAE [g]" if kind == "reg" else "ROC-AUC")


if __name__ == "__main__":
    res = Parallel(n_jobs=-1)(delayed(job)(t, c, s) for t in TARGETS for c in CONFIGS for s in range(5))
    out = (pd.DataFrame(res).groupby(["target", "config", "metric"], sort=False)
           .agg(n_features=("n_features", "first"), mean=("score", "mean"), std=("score", "std")).reset_index())
    out.to_csv(RESULTS_DIR / "tables" / "phase4_logger_check.csv", index=False)
    print(out.round(3).to_string(index=False))
