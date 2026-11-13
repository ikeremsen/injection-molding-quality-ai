"""Models, validation splits and metrics shared by Phase 2-4.

Rules this file enforces:
- Every preprocessing step (imputer, scaler) sits inside a Pipeline, so it is
  fitted on the training rows only.
- The decision threshold is tuned with an inner grouped CV on the training
  rows only. The test rows are never used to tune anything.
- Splits are always by group (day + experiment), never by row.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, mean_absolute_error, r2_score, roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEEDS = [0, 1, 2, 3, 4]


def _imp():
    return SimpleImputer(strategy="median", keep_empty_features=True)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
CLASSIFIERS = ["majority", "stump", "logreg", "rf", "gb"]
REGRESSORS = ["mean", "single_linear", "ridge", "rf", "gb"]


def make_classifier(name: str, seed: int):
    if name == "majority":      # always predicts the training NOK rate
        return DummyClassifier(strategy="prior")
    if name == "stump":         # ONE feature, ONE threshold, both chosen on training data
        return make_pipeline(_imp(), DecisionTreeClassifier(max_depth=1, class_weight="balanced", random_state=seed))
    if name == "logreg":
        return make_pipeline(_imp(), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    if name == "rf":
        return make_pipeline(_imp(), RandomForestClassifier(
            n_estimators=200, min_samples_leaf=3, class_weight="balanced_subsample",
            n_jobs=1, random_state=seed))
    if name == "gb":
        return make_pipeline(_imp(), HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=seed))
    raise ValueError(name)


class SingleFeatureLinear(BaseEstimator, RegressorMixin):
    """Regression baseline: straight line on the one feature most correlated with y (in training)."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        sd = X.std(axis=0)
        corr = np.zeros(X.shape[1])
        ok = sd > 0
        corr[ok] = np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in np.flatnonzero(ok)])
        self.j_ = int(np.nanargmax(corr))
        self.lr_ = LinearRegression().fit(X[:, [self.j_]], y)
        return self

    def predict(self, X):
        return self.lr_.predict(np.asarray(X, dtype=float)[:, [self.j_]])


def make_regressor(name: str, seed: int):
    if name == "mean":
        return DummyRegressor(strategy="mean")
    if name == "single_linear":
        return make_pipeline(_imp(), SingleFeatureLinear())
    if name == "ridge":
        return make_pipeline(_imp(), StandardScaler(), Ridge(alpha=10.0))
    if name == "rf":
        return make_pipeline(_imp(), RandomForestRegressor(n_estimators=200, min_samples_leaf=3,
                                                           n_jobs=1, random_state=seed))
    if name == "gb":
        return make_pipeline(_imp(), HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                                   max_iter=300, random_state=seed))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# Threshold tuning (training data only)
# ---------------------------------------------------------------------------
def best_threshold(y, p) -> float:
    """Threshold that maximises balanced accuracy on (out-of-fold) training predictions."""
    cands = np.unique(np.round(p, 4))
    if len(cands) < 2:
        return 0.5
    scores = [balanced_accuracy_score(y, p >= t) for t in cands]
    return float(cands[int(np.argmax(scores))])


def tuned_threshold(model, X, y, groups, seed: int, n_inner: int = 3) -> float:
    """Inner grouped CV on the TRAINING rows -> out-of-fold probabilities -> threshold."""
    n_pos_groups = pd.Series(y).groupby(np.asarray(groups)).max().sum()
    k = int(min(n_inner, n_pos_groups))
    if k < 2:
        return 0.5
    oof = np.full(len(y), np.nan)
    for tr, va in StratifiedGroupKFold(k, shuffle=True, random_state=seed).split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        m = clone(model).fit(X.iloc[tr], y[tr])
        oof[va] = m.predict_proba(X.iloc[va])[:, 1]
    ok = ~np.isnan(oof)
    return best_threshold(y[ok], oof[ok]) if ok.sum() and len(np.unique(y[ok])) == 2 else 0.5


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def classification_metrics(y, p, yhat_tuned, groups) -> dict:
    """y: true 0/1, p: predicted probability, yhat_tuned: 0/1 using the training-tuned threshold."""
    out = dict(n=len(y), n_pos=int(y.sum()),
               n_pos_groups=int(pd.Series(y).groupby(np.asarray(groups)).max().sum()),
               n_groups=int(pd.Series(groups).nunique()))
    if len(np.unique(y)) < 2:   # e.g. a test day without any NOK part
        out.update(roc_auc=np.nan, pr_auc=np.nan, group_auc=np.nan,
                   bal_acc_05=np.nan, bal_acc_tuned=np.nan, note="not computable: one class in test")
        return out
    g = pd.DataFrame({"g": groups, "y": y, "p": p}).groupby("g").mean()
    tn, fp, fn, tp = confusion_matrix(y, yhat_tuned, labels=[0, 1]).ravel()
    out.update(roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
               group_auc=roc_auc_score(g["y"].round(), g["p"]),
               bal_acc_05=balanced_accuracy_score(y, p >= 0.5),
               bal_acc_tuned=balanced_accuracy_score(y, yhat_tuned),
               tn=tn, fp=fp, fn=fn, tp=tp, note="")
    return out


def regression_metrics(y, pred, days) -> dict:
    out = dict(n=len(y), mae=mean_absolute_error(y, pred), r2=r2_score(y, pred))
    per_day = {}
    for d in np.unique(days):
        m = days == d
        per_day[d] = (mean_absolute_error(y[m], pred[m]), r2_score(y[m], pred[m]) if m.sum() > 2 else np.nan)
        out[f"mae_{d}"], out[f"r2_{d}"] = per_day[d]
    out["r2_within_day_mean"] = np.nanmean([v[1] for v in per_day.values()])
    return out


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def drop_unavailable(cols, X_train, X_test):
    """Cross-day schemes: drop columns that are completely missing on the train or test side.

    This is about which sensors exist on a day (known before any measurement),
    not about labels, so it does not leak test outcomes.
    """
    return [c for c in cols if X_train[c].notna().any() and X_test[c].notna().any()]


def cross_day_splits(df: pd.DataFrame):
    """Yields (scheme, test_name, train_idx, test_idx)."""
    # Note: "train on PP, test on ABS" is exactly the leave-one-day-out split with
    # the ABS day (2021-12-15) as test day, because ABS was run on one day only.
    days = df["day"].values
    for d in sorted(np.unique(days)):
        yield "leave_one_day_out", d, np.flatnonzero(days != d), np.flatnonzero(days == d)
