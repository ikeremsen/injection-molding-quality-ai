"""Phase 4c: Polenta waste vs ACCEPTABLE only (classes 1 and 2).

Why: Phase 4 found that cycle time and fill time split the data into two machine
regimes (waste+acceptable at ~74.8 s, target+inefficient at ~75.7 s with other
fill times). So "waste vs rest" is mostly "waste vs acceptable" plus an easy
regime check. This script isolates the hard, production-relevant part: two
neighbouring classes made in the same regime.

Run from `05 Code`:  python scripts/phase4_polenta_waste_vs_acceptable.py
Writes results/tables/phase4_polenta_waste_vs_acceptable.csv
"""
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ns = runpy.run_path(str(Path(__file__).with_name("phase4_polenta.py")), run_name="lib")
raw, FEATURES, SUBSETS, CH_COLS = ns["raw"], ns["FEATURES"], ns["SUBSETS"], ns["CH_COLS"]
within_class_chunks, rf = ns["within_class_chunks"], ns["rf"]
from src.loaders import RESULTS_DIR  # noqa: E402

SETS = {
    "all 13": FEATURES,
    "without phase timing (cycle time, fill time)": [f for f in FEATURES if f not in CH_COLS["phase_timing"]],
    "melt pressure only (2 features)": CH_COLS["melt_pressure"],
    "injection pressure peak only": ["APVs - Specific injection pressure peak value"],
    "phase timing only": CH_COLS["phase_timing"],
}
rows = []
for subset, mask in SUBSETS.items():
    d = raw[mask & raw.quality.isin([1, 2]).values].reset_index(drop=True)
    y = (d.quality == 1).astype(int).values
    for name, cols in SETS.items():
        for seed in [0, 1, 2]:
            p = np.zeros(len(y))
            for tr, te in within_class_chunks(y, 5):
                p[te] = rf(seed).fit(d.iloc[tr][cols], y[tr]).predict_proba(d.iloc[te][cols])[:, 1]
            rows.append(dict(subset=subset, features=name, seed=seed, n_waste=int(y.sum()), n_acceptable=int((1 - y).sum()),
                             roc_auc=roc_auc_score(y, p), bal_acc_05=balanced_accuracy_score(y, p >= 0.5)))
out = pd.DataFrame(rows).groupby(["subset", "features"], sort=False).agg(
    n_waste=("n_waste", "first"), n_acceptable=("n_acceptable", "first"),
    roc_auc=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"), bal_acc_05=("bal_acc_05", "mean")).reset_index()
out.to_csv(RESULTS_DIR / "tables" / "phase4_polenta_waste_vs_acceptable.csv", index=False)
print(out.round(3).to_string(index=False))
