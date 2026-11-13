"""Phase 4a: which SKZ signals matter? Importance per channel, under grouped CV.

Run from `05 Code`:   python scripts/phase4_skz.py
Writes results/tables/phase4_skz_importance.csv, phase4_skz_shap_features.csv,
       phase4_skz_selection.csv

Three complementary views, all computed on TEST folds of a grouped CV:
1. Channel permutation importance: shuffle all features of one channel
   together and measure how much the test score drops. Shuffling happens only
   among parts of the SAME DAY, so a channel cannot look important just because
   it tells the days apart.
2. SHAP: how much each feature moves the individual predictions (mean |SHAP|),
   summed per channel.
3. Nested forward selection: inside each training fold, add channels one at a
   time (the one that helps most on an inner grouped CV), then score the chosen
   set on the untouched outer test fold. This answers "how few channels do we
   need?" without the optimism of choosing and scoring on the same data.

Weight is scored by an "offset-corrected MAE": the mean error per test day is
removed before taking |error|. This assumes the daily level is recalibrated (for
example with one weighed reference part), which Phase 2 showed is necessary
anyway, and keeps the material / day level from dominating the importance.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR  # noqa: E402
from src.modeling import _imp, make_classifier, make_regressor  # noqa: E402
from src.signals import add_previous_cycle_dosing, channel_table, skz_channel  # noqa: E402
from src.skz_features import load_skz_model_table  # noqa: E402

OUT = RESULTS_DIR / "tables"
TARGETS = {"LBL_NOK": "cls", "LBL_SinkMarks": "cls", "LBL_Underfilled": "cls", "SCA_PartWeight": "reg"}
IMP_MODELS = {"cls": ["logreg", "rf"], "reg": ["ridge", "rf"]}
IMP_SEEDS = [0, 1, 2]
SEL_SEEDS = [0, 1, 2]
N_REPEATS = 10
MAX_CHANNELS = 5

df, FS = load_skz_model_table()
df = add_previous_cycle_dosing(df)
# Use the dose that actually fed this shot (previous cycle), not the next one.
COLS = [c for c in FS["L1_settings"] + FS["L4_curves"] if c not in ("CALC_DosedWeight", "CALC_DosingStroke")]
COLS += ["CALC_DosedWeightPrev", "CALC_DosingStrokePrev"]
CHMAP = channel_table(COLS, skz_channel)
CHANNELS = sorted(CHMAP.channel.unique())
CH_COLS = {ch: CHMAP.loc[CHMAP.channel == ch, "feature"].tolist() for ch in CHANNELS}
X_ALL = df[COLS]
G, DAY = df["group"].values, df["day"].values


# ---------------------------------------------------------------------------
def score(kind, y, pred, days):
    """Higher = better for both kinds (MAE is returned negated)."""
    if kind == "cls":
        return roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan
    res = pred - y
    for d in np.unique(days):              # remove the mean error of each day (daily recalibration)
        m = days == d
        res[m] = res[m] - res[m].mean()
    return -np.abs(res).mean()


def splits(kind, y, seed, n=5, groups=G, X=None):
    X = np.zeros(len(y)) if X is None else X
    if kind == "cls":
        return StratifiedGroupKFold(n, shuffle=True, random_state=seed).split(X, y, groups)
    return GroupKFold(n, shuffle=True, random_state=seed).split(X, y, groups)


def build(kind, model_name, seed):
    return make_classifier(model_name, seed) if kind == "cls" else make_regressor(model_name, seed)


def predict(kind, model, X):
    return model.predict_proba(X)[:, 1] if kind == "cls" else model.predict(X)


def permute_within_day(X, cols, days, rng):
    Xp = X.copy()
    for d in np.unique(days):
        idx = np.flatnonzero(days == d)
        Xp.iloc[idx, [X.columns.get_loc(c) for c in cols]] = X.iloc[idx][cols].values[rng.permutation(len(idx))]
    return Xp


def shap_values(kind, model_name, model, X_tr, X_te):
    """Mean |SHAP| per feature on the test rows."""
    prep, est = model[:-1], model[-1]
    Xt_te = prep.transform(X_te)
    if model_name == "rf":
        sv = shap.TreeExplainer(est).shap_values(Xt_te, check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:                     # classifier: (rows, features, classes) -> class 1
            sv = sv[..., 1]
    else:                                    # linear models: exact SHAP = coef * (x - mean)
        masker = shap.maskers.Independent(prep.transform(X_tr), max_samples=200)
        sv = shap.LinearExplainer(est, masker).shap_values(Xt_te)
    return np.abs(sv).mean(axis=0)


# ---------------------------------------------------------------------------
# 1 + 2: permutation and SHAP
# ---------------------------------------------------------------------------
def importance_job(target, model_name, seed):
    kind = TARGETS[target]
    y = df[target].astype(float).values if kind == "reg" else df[target].astype(int).values
    rng = np.random.default_rng(seed)
    rows, shap_rows = [], []
    for fold, (tr, te) in enumerate(splits(kind, y, seed)):
        model = build(kind, model_name, seed).fit(X_ALL.iloc[tr], y[tr])
        X_te = X_ALL.iloc[te]
        base = score(kind, y[te], predict(kind, model, X_te), DAY[te])
        if np.isnan(base):
            continue
        for ch in CHANNELS:
            drops = [base - score(kind, y[te], predict(kind, model, permute_within_day(X_te, CH_COLS[ch], DAY[te], rng)),
                                  DAY[te]) for _ in range(N_REPEATS)]
            rows.append(dict(target=target, model=model_name, seed=seed, fold=fold, channel=ch,
                             base_score=base, perm_drop=np.mean(drops)))
        sv = shap_values(kind, model_name, model, X_ALL.iloc[tr], X_te)
        shap_rows.append(pd.DataFrame(dict(target=target, model=model_name, seed=seed, fold=fold,
                                           feature=COLS, mean_abs_shap=sv)))
    return pd.DataFrame(rows), pd.concat(shap_rows)


# ---------------------------------------------------------------------------
# 3: nested forward selection of channels
# ---------------------------------------------------------------------------
def selection_model(kind, seed):
    if kind == "cls":
        return make_classifier("logreg", seed)           # best SKZ classifier in Phase 2
    return make_pipeline(_imp(), RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                                       n_jobs=1, random_state=seed))


def cv_score(kind, cols, idx, y, seed):
    """Pooled out-of-fold score of `cols` on the rows `idx` (inner grouped 3-fold CV)."""
    pred = np.full(len(idx), np.nan)
    yy, gg = y[idx], G[idx]
    for a, b in splits(kind, yy, seed, n=3, groups=gg):
        if kind == "cls" and len(np.unique(yy[a])) < 2:
            continue
        m = selection_model(kind, seed).fit(X_ALL.iloc[idx[a]][cols], yy[a])
        pred[b] = predict(kind, m, X_ALL.iloc[idx[b]][cols])
    ok = ~np.isnan(pred)
    return score(kind, yy[ok], pred[ok], DAY[idx][ok])


def selection_job(target, variant, seed):
    kind = TARGETS[target]
    y = df[target].astype(float).values if kind == "reg" else df[target].astype(int).values
    candidates = [c for c in CHANNELS if not (variant == "measured signals only" and c == "setpoints")]
    pooled = {k: np.full(len(y), np.nan) for k in range(1, MAX_CHANNELS + 1)}
    pooled["all"] = np.full(len(y), np.nan)
    rows = []
    for fold, (tr, te) in enumerate(splits(kind, y, seed)):
        chosen = []
        for k in range(1, MAX_CHANNELS + 1):
            inner = {ch: cv_score(kind, sum((CH_COLS[c] for c in chosen + [ch]), []), tr, y, seed)
                     for ch in candidates if ch not in chosen}
            best = max(inner, key=lambda c: -np.inf if np.isnan(inner[c]) else inner[c])
            chosen.append(best)
            cols = sum((CH_COLS[c] for c in chosen), [])
            m = selection_model(kind, seed).fit(X_ALL.iloc[tr][cols], y[tr])
            pooled[k][te] = predict(kind, m, X_ALL.iloc[te][cols])
            rows.append(dict(target=target, variant=variant, seed=seed, fold=fold, k=k, added=best,
                             inner_score=inner[best]))
        cols = sum((CH_COLS[c] for c in candidates), [])
        m = selection_model(kind, seed).fit(X_ALL.iloc[tr][cols], y[tr])
        pooled["all"][te] = predict(kind, m, X_ALL.iloc[te][cols])
    out = pd.DataFrame(rows)
    summ = [dict(target=target, variant=variant, seed=seed, k=k,
                 outer_score=score(kind, y, pooled[k], DAY)) for k in pooled]
    return out, pd.DataFrame(summ)


if __name__ == "__main__":
    t0 = time.time()
    print("channels:", {c: len(v) for c, v in CH_COLS.items()})
    imp_tasks = [delayed(importance_job)(t, m, s) for t, kind in TARGETS.items()
                 for m in IMP_MODELS[kind] for s in IMP_SEEDS]
    sel_tasks = [delayed(selection_job)(t, v, s) for t in TARGETS
                 for v in ["with setpoints", "measured signals only"] for s in SEL_SEEDS]
    res = Parallel(n_jobs=-1, verbose=5)(imp_tasks + sel_tasks)
    imp, shp = zip(*res[:len(imp_tasks)])
    sel, sel_sum = zip(*res[len(imp_tasks):])
    pd.concat(imp).to_csv(OUT / "phase4_skz_importance.csv", index=False)
    s = pd.concat(shp).merge(CHMAP, on="feature")
    s.to_csv(OUT / "phase4_skz_shap_features.csv", index=False)
    pd.concat(sel).to_csv(OUT / "phase4_skz_selection_steps.csv", index=False)
    pd.concat(sel_sum).to_csv(OUT / "phase4_skz_selection_scores.csv", index=False)
    CHMAP.to_csv(OUT / "phase4_skz_channel_map.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
