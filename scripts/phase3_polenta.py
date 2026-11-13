"""Phase 3: production check on the Polenta lens data. Models x feature sets x row subsets x CV schemes.

Run from `05 Code`:
    python scripts/phase3_polenta.py
Writes results/tables/phase3_runs.csv, phase3_summary.csv, phase3_confusion.csv.

Validation (agreed after Phase 1):
- The file is sorted by class (370 waste, then 406 acceptable, ...), so plain
  contiguous blocks would put mostly one class in each test fold.
- "within-class chunks": every class is cut into 5 contiguous chunks in file
  order; test fold k = chunk k of every class. Neighbouring rows (probably made
  minutes apart) stay on the same side, and every fold contains every class.
- "random rows (optimistic reference)": ordinary shuffled stratified 5-fold.
  Only shown to measure how much neighbouring rows inflate the score.

Row subsets:
- "all rows"
- "clean rows": rows without any value having more than 4 decimals. Such values
  cannot come from the MES (it logs 1-3 decimals) and occur mostly in the waste
  class (75 % of waste rows), so they may be synthetic or processed.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR, load_polenta  # noqa: E402
from src.modeling import CLASSIFIERS, SEEDS, best_threshold, make_classifier  # noqa: E402

OUT = RESULTS_DIR / "tables"
CLASS_NAMES = {1: "waste", 2: "acceptable", 3: "target", 4: "inefficient"}

raw = load_polenta()
ALL_FEATURES = [c for c in raw.columns if c != "quality"]
# "Basic" logger: only temperatures, times and screw-volume values, i.e. no
# pressure, force or torque sensors. This answers a hardware question:
# do we need the pressure/force/torque channels?
BASIC_6 = ["Melt temperature", "Mold temperature", "time_to_fill", "ZDx - Plasticizing time",
           "ZUx - Cycle time", "SVo - Shot volume"]
FEATURE_SETS = {"all_13": ALL_FEATURES, "basic_6": BASIC_6}


def n_decimals(v: float) -> int:
    s = repr(float(v))
    return 0 if "." not in s else len(s.split(".")[1].rstrip("0"))


LONG_DEC = raw[ALL_FEATURES].map(lambda v: n_decimals(v) > 4).any(axis=1).values
SUBSETS = {"all rows": np.ones(len(raw), bool), "clean rows": ~LONG_DEC}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def within_class_chunks(y: np.ndarray, n_splits: int):
    """Yield (train_idx, test_idx). y must be in file order."""
    chunk_id = np.empty(len(y), int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)                     # rows of this class, in file order
        for k, part in enumerate(np.array_split(idx, n_splits)):
            chunk_id[part] = k
    for k in range(n_splits):
        yield np.flatnonzero(chunk_id != k), np.flatnonzero(chunk_id == k)


def random_rows(y: np.ndarray, n_splits: int, seed: int):
    yield from StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)


# ---------------------------------------------------------------------------
# One experiment = (task, feature set, row subset, scheme, model, seed)
# ---------------------------------------------------------------------------
def run(task, fs, subset, scheme, model_name, seed):
    d = raw[SUBSETS[subset]].reset_index(drop=True)      # keeps file order
    X = d[FEATURE_SETS[fs]]
    orig = d["quality"].values
    y = (orig == 1).astype(int) if task == "waste_vs_rest" else orig
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    yhat = np.zeros(len(y), int)
    fold_acc, fold_bacc = [], []
    splits = (within_class_chunks(y, 5) if scheme == "within-class chunks" else random_rows(y, 5, seed))
    for tr, te in splits:
        model = make_classifier(model_name, seed).fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[te])
        proba[te] = p
        if task == "waste_vs_rest":
            thr = 0.5 if model_name == "majority" else inner_threshold(model, X.iloc[tr], y[tr], scheme, seed)
            yhat[te] = (p[:, 1] >= thr).astype(int)
        else:
            yhat[te] = classes[p.argmax(axis=1)]
        fold_acc.append(accuracy_score(y[te], yhat[te]))
        fold_bacc.append(balanced_accuracy_score(y[te], yhat[te]))

    row = dict(task=task, features=fs, subset=subset, scheme=scheme, model=model_name, seed=seed,
               n=len(y), accuracy=accuracy_score(y, yhat), bal_acc=balanced_accuracy_score(y, yhat),
               fold_acc_std=np.std(fold_acc), fold_bacc_min=np.min(fold_bacc),
               majority_rate=pd.Series(y).value_counts(normalize=True).max())
    if task == "waste_vs_rest":
        row.update(roc_auc=roc_auc_score(y, proba[:, 1]), pr_auc=average_precision_score(y, proba[:, 1]),
                   waste_rate=y.mean(),
                   # untuned reference: the plain 0.5 cut, to show what the inner threshold tuning costs or gains
                   bal_acc_05=balanced_accuracy_score(y, proba[:, 1] >= 0.5),
                   accuracy_05=accuracy_score(y, proba[:, 1] >= 0.5),
                   # false alarms split by the TRUE class (which good lenses get called waste?)
                   **{f"false_alarm_rate_{CLASS_NAMES[c]}": float((proba[orig == c, 1] >= 0.5).mean())
                      for c in (2, 3, 4)})
    else:
        row["roc_auc"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")
    cm = confusion_matrix(y, yhat, labels=classes)
    return row, dict(task=task, features=fs, subset=subset, scheme=scheme, model=model_name, seed=seed,
                     labels=list(classes), cm=cm.tolist())


def inner_threshold(model, X, y, scheme, seed):
    """Threshold from an inner CV on the TRAINING rows, split the same way as the outer CV."""
    oof = np.zeros(len(y))
    inner = within_class_chunks(y, 3) if scheme == "within-class chunks" else random_rows(y, 3, seed)
    for tr, va in inner:
        oof[va] = clone(model).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[va])[:, 1]
    return best_threshold(y, oof)


if __name__ == "__main__":
    t0 = time.time()
    print(f"clean rows: {SUBSETS['clean rows'].sum()} of {len(raw)}; by class:",
          raw[SUBSETS['clean rows']]['quality'].value_counts().sort_index().to_dict())
    tasks = [delayed(run)(task, fs, sub, sch, m, s)
             for task in ["4_class", "waste_vs_rest"]
             for fs in FEATURE_SETS for sub in SUBSETS
             for sch in ["within-class chunks", "random rows (optimistic reference)"]
             for m in CLASSIFIERS for s in SEEDS]
    out = Parallel(n_jobs=-1)(tasks)
    runs = pd.DataFrame([o[0] for o in out])
    runs.to_csv(OUT / "phase3_runs.csv", index=False)
    pd.DataFrame([o[1] for o in out]).to_csv(OUT / "phase3_confusion.csv", index=False)
    keys = ["task", "features", "subset", "scheme", "model"]
    metrics = ["accuracy", "bal_acc", "roc_auc", "pr_auc", "fold_acc_std", "fold_bacc_min",
               "accuracy_05", "bal_acc_05", "false_alarm_rate_acceptable", "false_alarm_rate_target",
               "false_alarm_rate_inefficient"]
    summ = runs.groupby(keys, sort=False).agg(
        **{f"{m}_mean": (m, "mean") for m in metrics},
        **{f"{m}_std_seeds": (m, "std") for m in metrics[:4]},
        n=("n", "first"), majority_rate=("majority_rate", "first")).reset_index()
    summ.to_csv(OUT / "phase3_summary.csv", index=False)
    print(f"done in {time.time() - t0:.0f} s")
