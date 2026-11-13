"""Phase 2a: SKZ defect classification. Feature ladder x models x validation schemes.

Run from `05 Code`:
    python scripts/phase2_skz_classification.py
Writes:
    results/tables/phase2_cls_runs.csv     one row per (target, scheme, test, features, model, seed)
    results/tables/phase2_cls_summary.csv  mean and std over seeds
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR  # noqa: E402
from src.modeling import (CLASSIFIERS, SEEDS, classification_metrics,  # noqa: E402
                          cross_day_splits, drop_unavailable, make_classifier, tuned_threshold)
from src.skz_features import load_skz_model_table  # noqa: E402

TARGETS = ["LBL_NOK", "LBL_SinkMarks", "LBL_Underfilled"]
CROSS_DAY_SEEDS = SEEDS[:3]   # only model randomness changes between seeds here (the split is fixed)
OUT = RESULTS_DIR / "tables"
OUT.mkdir(parents=True, exist_ok=True)

df, FEATURE_SETS = load_skz_model_table()
GROUPS = df["group"].values


def fit_predict(model_name, seed, X_tr, y_tr, g_tr, X_te):
    """Fit on training rows, return test probabilities, test 0/1 at the train-tuned threshold, stump feature."""
    model = make_classifier(model_name, seed)
    if len(np.unique(y_tr)) < 2:     # no positives in training: nothing can be learned
        p = np.full(len(X_te), float(y_tr.mean()))
        return p, (p >= 0.5).astype(int), None
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    thr = 0.5 if model_name == "majority" else tuned_threshold(model, X_tr, y_tr, g_tr, seed)
    stump_feat = None
    if model_name == "stump":
        j = model[-1].tree_.feature[0]
        stump_feat = X_tr.columns[j] if j >= 0 else None
    return p, (p >= thr).astype(int), stump_feat


def run_group_cv(target, fs, model_name, seed):
    cols = FEATURE_SETS[fs]
    X, y = df[cols], df[target].astype(int).values
    p_all, yhat_all, stumps = np.zeros(len(y)), np.zeros(len(y), int), []
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, GROUPS):
        p, yhat, sf = fit_predict(model_name, seed, X.iloc[tr], y[tr], GROUPS[tr], X.iloc[te])
        p_all[te], yhat_all[te] = p, yhat
        stumps.append(sf)
    row = classification_metrics(y, p_all, yhat_all, GROUPS)
    row.update(target=target, scheme="group_5fold", test="all (pooled out-of-fold)", features=fs,
               model=model_name, seed=seed, n_features=len(cols),
               stump_feature=pd.Series(stumps).mode().iloc[0] if any(stumps) else None)
    return [row]


def run_cross_day(target, fs, model_name, seed):
    rows = []
    for scheme, test_name, tr, te in cross_day_splits(df):
        cols = drop_unavailable(FEATURE_SETS[fs], df.iloc[tr], df.iloc[te])
        X, y = df[cols], df[target].astype(int).values
        p, yhat, sf = fit_predict(model_name, seed, X.iloc[tr], y[tr], GROUPS[tr], X.iloc[te])
        row = classification_metrics(y[te], p, yhat, GROUPS[te])
        row.update(target=target, scheme=scheme, test=test_name, features=fs, model=model_name,
                   seed=seed, n_features=len(cols), stump_feature=sf,
                   n_pos_groups_train=int(pd.Series(y[tr]).groupby(GROUPS[tr]).max().sum()))
        rows.append(row)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    tasks = []
    for target in TARGETS:
        for fs in FEATURE_SETS:
            for m in CLASSIFIERS:
                tasks += [delayed(run_group_cv)(target, fs, m, s) for s in SEEDS]
                tasks += [delayed(run_cross_day)(target, fs, m, s) for s in CROSS_DAY_SEEDS]
    print(f"{len(tasks)} tasks")
    results = Parallel(n_jobs=-1, verbose=5)(tasks)
    runs = pd.DataFrame([r for rows in results for r in rows])
    runs.to_csv(OUT / "phase2_cls_runs.csv", index=False)

    keys = ["target", "scheme", "test", "features", "model"]
    metrics = ["roc_auc", "pr_auc", "group_auc", "bal_acc_05", "bal_acc_tuned"]
    summ = runs.groupby(keys, sort=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std": (m, "std") for m in metrics},
        tn=("tn", "mean"), fp=("fp", "mean"), fn=("fn", "mean"), tp=("tp", "mean"),
        n_pos_groups=("n_pos_groups", "first"), n_groups=("n_groups", "first"),
        n_features=("n_features", "first"), note=("note", "first"),
        stump_feature=("stump_feature", lambda s: s.mode().iloc[0] if s.notna().any() else None),
    ).reset_index()
    summ.to_csv(OUT / "phase2_cls_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
