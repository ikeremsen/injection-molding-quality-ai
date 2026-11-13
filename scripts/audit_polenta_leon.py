"""Phase 1 audit for the Polenta and León datasets. No modeling.

Run from the `05 Code` folder:
    python scripts/audit_polenta_leon.py
Writes text reports to results/audit/.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loaders import RESULTS_DIR, load_leon, load_polenta  # noqa: E402

OUT = RESULTS_DIR / "audit"
OUT.mkdir(parents=True, exist_ok=True)


def n_decimals(x: float) -> int:
    """How many decimals a number was written with (repr-based)."""
    s = repr(float(x))
    return 0 if "." not in s else len(s.split(".")[1].rstrip("0"))


def basic_table(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_missing": df.isna().sum(),
        "n_unique": df.nunique(),
        "min": df.min(numeric_only=True),
        "median": df.median(numeric_only=True),
        "max": df.max(numeric_only=True),
    })


# ---------------------------------------------------------------------------
# Polenta
# ---------------------------------------------------------------------------
def audit_polenta(lines: list[str]) -> None:
    df = load_polenta()
    feats = [c for c in df.columns if c != "quality"]
    lines.append("=== POLENTA (lenses) ===")
    lines.append(f"shape: {df.shape}")
    lines.append(f"exact duplicate rows: {df.duplicated().sum()}")
    lines.append("\ncolumn summary:\n" + basic_table(df).to_string())
    lines.append("\nclass counts (1 waste, 2 acceptable, 3 target, 4 inefficient):")
    lines.append(df["quality"].value_counts().sort_index().to_string())

    # Long decimals = value not copied from the machine/MES (which logs 1-3 decimals).
    # Could indicate interpolation, imputation or synthetic oversampling (e.g. SMOTE).
    long_dec = df[feats].map(lambda v: n_decimals(v) > 4)
    df["_n_long_decimal_cells"] = long_dec.sum(axis=1)
    lines.append("\nfraction of cells with >4 decimals, per column:")
    lines.append(long_dec.mean().round(3).to_string())
    lines.append("\nrows with >=1 long-decimal cell, by class:")
    tab = (df.assign(has_long=df["_n_long_decimal_cells"] > 0)
             .groupby("quality")["has_long"].agg(["sum", "mean"]))
    lines.append(tab.round(3).to_string())

    # Is the label ordered in the file (blocks of one class)? Count label changes.
    changes = (df["quality"].diff() != 0).sum() - 1
    lines.append(f"\nnumber of label changes along row order: {changes} "
                 f"(a shuffled file would show ~{int(len(df) * 0.72)})")
    # Show run lengths of the label sequence
    runs = (df["quality"] != df["quality"].shift()).cumsum()
    run_tab = df.groupby(runs)["quality"].agg(["first", "size"])
    lines.append("label runs along row order (class, length):")
    lines.append(", ".join(f"{a}x{b}" for a, b in run_tab.values))

    # Where does row order show jumps that may be day boundaries? Mold temperature
    # and cycle time are slow-changing machine states, so large jumps hint at a
    # new production day or a setting change.
    for c in ["Mold temperature", "ZUx - Cycle time", "SVo - Shot volume"]:
        jumps = df[c].diff().abs()
        top = jumps.nlargest(8).sort_index()
        lines.append(f"\nlargest row-to-row jumps in '{c}' (row index: jump):")
        lines.append(", ".join(f"{i}: {v:.2f}" for i, v in top.items()))
    lines.append("")


# ---------------------------------------------------------------------------
# León
# ---------------------------------------------------------------------------
def audit_leon(lines: list[str]) -> None:
    df = load_leon()
    settings = ["Tinj", "tinj", "Pinj", "Ph", "Bp", "th"]
    lines.append("=== LEON PP (batches) ===")
    lines.append(f"shape: {df.shape}")
    lines.append(f"columns: {list(df.columns)}")
    lines.append("\ncolumn summary:\n" + basic_table(df).to_string())
    lines.append("\nlevels of each setting:")
    for c in settings:
        lines.append(f"  {c}: {sorted(df[c].unique().tolist())}")
    n_combo = df[settings].drop_duplicates().shape[0]
    full = int(np.prod([df[c].nunique() for c in settings]))
    lines.append(f"unique setting combinations: {n_combo} (full factorial would be {full})")
    lines.append("\nviability label counts:\n" + df["label_viability"].value_counts().to_string())
    lines.append("\nmissing outputs by viability label:")
    outs = [c for c in df.columns if c.endswith("_weight") or c in ("weight_avg", "energy", "cycle_time")]
    lines.append(df.groupby("label_viability")[outs].apply(lambda g: g.isna().mean()).round(2).to_string())

    # Checks: is weight_avg the mean of P1..P5? Is cycle time a pure function of settings?
    w = df[[f"P{i}_weight" for i in range(1, 6)]]
    diff = (w.mean(axis=1) - df["weight_avg"]).abs()
    lines.append(f"\nmax |mean(P1..P5) - weight_avg|: {diff.max():.4f}")
    ct_unique = df.groupby(settings)["cycle_time"].nunique().max()
    lines.append(f"max distinct cycle_time values per setting combination: {ct_unique}")
    for c in settings:
        lines.append(f"  cycle_time spread by {c}: "
                     + df.groupby(c)["cycle_time"].mean().round(2).to_dict().__str__())
    # Spread of part weights inside a batch (part-to-part noise)
    lines.append(f"\nwithin-batch weight std (median over batches): {w.std(axis=1).median():.4f} g")
    lines.append(f"between-batch weight_avg std: {df['weight_avg'].std():.4f} g")
    lines.append("\nenergy: non-missing rows = "
                 f"{df['energy'].notna().sum()}; by label "
                 + df.groupby('label_viability')['energy'].count().to_dict().__str__())
    lines.append("")


if __name__ == "__main__":
    lines: list[str] = []
    audit_polenta(lines)
    audit_leon(lines)
    text = "\n".join(lines)
    (OUT / "audit_polenta_leon.txt").write_text(text, encoding="utf-8")
    print(text)
