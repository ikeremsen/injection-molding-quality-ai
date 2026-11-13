"""Phase 4b: which Polenta signals matter? Same three views as phase4_skz.py.

Run from `05 Code`:   python scripts/phase4_polenta.py
Writes results/tables/phase4_polenta_importance.csv, phase4_polenta_shap.csv,
       phase4_polenta_selection_steps.csv, phase4_polenta_selection_scores.csv

- Model: random forest (best Polenta model in Phase 3).
- CV: within-class chunks (5 outer, 3 inner), as agreed after Phase 1.
- Main subset: clean rows (the long-decimal rows removed); all rows for comparison.
- Scores: waste vs rest -> ROC-AUC; 4 classes -> macro one-vs-rest ROC-AUC
  (smoother than accuracy, so small importance differences are visible).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR, load_polenta  # noqa: E402
from src.modeling import _imp  # noqa: E402
from src.signals import POLENTA_CHANNELS  # noqa: E402

OUT = RESULTS_DIR / "tables"
SEEDS = [0, 1, 2]
N_REPEATS = 10

raw = load_polenta()
FEATURES = [c for c in raw.columns if c != "quality"]
CHANNELS = sorted(set(POLENTA_CHANNELS.values()))
CH_COLS = {ch: [f for f in FEATURES if POLENTA_CHANNELS[f] == ch] for ch in CHANNELS}


def n_decimals(v):
    s = repr(float(v))
    return 0 if "." not in s else len(s.split(".")[1].rstrip("0"))


LONG_DEC = raw[FEATURES].map(lambda v: n_decimals(v) > 4).any(axis=1).values
SUBSETS = {"clean rows": ~LONG_DEC, "all rows": np.ones(len(raw), bool)}


def within_class_chunks(y, n):
    """Each class cut into n contiguous chunks in file order; fold k tests chunk k of every class."""
    chunk = np.empty(len(y), int)
    for c in np.unique(y):
        for k, part in enumerate(np.array_split(np.flatnonzero(y == c), n)):
            chunk[part] = k
    for k in range(n):
        yield np.flatnonzero(chunk != k), np.flatnonzero(chunk == k)


def rf(seed, n_estimators=200):
    return make_pipeline(_imp(), RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=3,
                                                        class_weight="balanced_subsample", n_jobs=1,
                                                        random_state=seed))


def score(task, y, proba):
    if task == "waste_vs_rest":
        return roc_auc_score(y, proba[:, 1])
    return roc_auc_score(y, proba, multi_class="ovr", average="macro")


def data(task, subset):
    d = raw[SUBSETS[subset]].reset_index(drop=True)
    y = (d["quality"] == 1).astype(int).values if task == "waste_vs_rest" else d["quality"].values
    return d[FEATURES], y


def importance_job(task, subset, seed):
    X, y = data(task, subset)
    rng = np.random.default_rng(seed)
    rows, shap_rows = [], []
    for fold, (tr, te) in enumerate(within_class_chunks(y, 5)):
        m = rf(seed).fit(X.iloc[tr], y[tr])
        Xte = X.iloc[te]
        base = score(task, y[te], m.predict_proba(Xte))

        def drop(cols):
            out = []
            for _ in range(N_REPEATS):
                Xp = Xte.copy()
                Xp[cols] = Xte[cols].values[rng.permutation(len(Xte))]   # channel columns shuffled together
                out.append(base - score(task, y[te], m.predict_proba(Xp)))
            return np.mean(out)

        for ch in CHANNELS:
            rows.append(dict(task=task, subset=subset, seed=seed, fold=fold, level="channel", name=ch,
                             perm_drop=drop(CH_COLS[ch])))
        for f in FEATURES:
            rows.append(dict(task=task, subset=subset, seed=seed, fold=fold, level="feature", name=f,
                             perm_drop=drop([f])))
        sv = np.asarray(shap.TreeExplainer(m[-1]).shap_values(m[:-1].transform(Xte), check_additivity=False))
        sv = np.abs(sv).mean(axis=0)                       # mean over test rows
        if sv.ndim == 2:                                   # (features, classes): average the classes
            sv = sv.mean(axis=1) if task == "4_class" else sv[:, 1]
        shap_rows.append(pd.DataFrame(dict(task=task, subset=subset, seed=seed, fold=fold,
                                           feature=FEATURES, mean_abs_shap=sv)))
    return pd.DataFrame(rows), pd.concat(shap_rows)


def selection_job(task, subset, seed, max_k=5):
    X, y = data(task, subset)
    n_cls = len(np.unique(y))
    pooled = {k: np.zeros((len(y), n_cls)) for k in list(range(1, max_k + 1)) + ["all"]}
    rows = []
    for fold, (tr, te) in enumerate(within_class_chunks(y, 5)):
        Xtr, ytr = X.iloc[tr].reset_index(drop=True), y[tr]

        def inner(cols):
            p = np.zeros((len(ytr), n_cls))
            for a, b in within_class_chunks(ytr, 3):
                p[b] = rf(seed, 100).fit(Xtr.iloc[a][cols], ytr[a]).predict_proba(Xtr.iloc[b][cols])
            return score(task, ytr, p)

        chosen = []
        for k in range(1, max_k + 1):
            sc = {ch: inner(sum((CH_COLS[c] for c in chosen + [ch]), [])) for ch in CHANNELS if ch not in chosen}
            best = max(sc, key=sc.get)
            chosen.append(best)
            cols = sum((CH_COLS[c] for c in chosen), [])
            pooled[k][te] = rf(seed, 100).fit(Xtr[cols], ytr).predict_proba(X.iloc[te][cols])
            rows.append(dict(task=task, subset=subset, seed=seed, fold=fold, k=k, added=best, inner_score=sc[best]))
        pooled["all"][te] = rf(seed, 100).fit(Xtr, ytr).predict_proba(X.iloc[te])
    summ = [dict(task=task, subset=subset, seed=seed, k=k, outer_score=score(task, y, p)) for k, p in pooled.items()]
    return pd.DataFrame(rows), pd.DataFrame(summ)


if __name__ == "__main__":
    t0 = time.time()
    combos = [(t, s) for t in ["waste_vs_rest", "4_class"] for s in SUBSETS]
    imp_tasks = [delayed(importance_job)(t, s, seed) for t, s in combos for seed in SEEDS]
    sel_tasks = [delayed(selection_job)(t, s, seed) for t, s in combos for seed in SEEDS]
    res = Parallel(n_jobs=-1, verbose=5)(imp_tasks + sel_tasks)
    imp, shp = zip(*res[:len(imp_tasks)])
    sel, summ = zip(*res[len(imp_tasks):])
    pd.concat(imp).to_csv(OUT / "phase4_polenta_importance.csv", index=False)
    pd.concat(shp).assign(channel=lambda d: d.feature.map(POLENTA_CHANNELS)).to_csv(
        OUT / "phase4_polenta_shap.csv", index=False)
    pd.concat(sel).to_csv(OUT / "phase4_polenta_selection_steps.csv", index=False)
    pd.concat(summ).to_csv(OUT / "phase4_polenta_selection_scores.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
