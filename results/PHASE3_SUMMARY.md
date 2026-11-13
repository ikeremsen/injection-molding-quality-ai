# Phase 3 summary: production check on the Polenta lens data

Scripts: `scripts/phase3_polenta.py` (≈12 min), `scripts/phase3_report.py`. Tables: `results/tables/phase3_tables.md`,
`phase3_summary.csv`, `phase3_runs.csv`, `phase3_confusion.csv`. Figures: `phase3_polenta_overview.png`, `phase3_polenta_confusion.png`.

## Setup
- **Data:** 1451 lenses, 13 machine values, 4 quality classes from the photometric uniformity U₀ (waste < 0.40 ≤ acceptable < 0.45 ≤ target ≤ 0.50 < inefficient).
- **Main validation, "within-class chunks":** each class is cut into 5 contiguous chunks in file order, and test fold k = chunk k of every class.
  "Random rows" (shuffled stratified 5-fold) is shown only as an **optimistic reference**.
- **Row subsets:** all rows, and **clean rows** (1122) without any value that has more than 4 decimals (the MES logs 1–3).
  Removing them leaves only **93 of 370 waste rows**.
- **Feature sets:** all 13, and **basic 6** = temperatures, times and shot volume only (no pressure, force or torque channels). The basic set asks a hardware question: are pressure sensors needed?
  Note: your earlier "6 simple features" may have been a different six, so the numbers are not directly comparable.
- **Models:** majority, single-feature threshold, logistic regression, random forest, gradient boosting; 5 seeds.
  For waste-vs-rest the threshold is tuned by an inner chunk-CV on the training rows; the plain 0.5 cut is reported as a second, untuned reference.

> **Correction added in Phase 4:** cycle time and fill time split the data into two machine regimes
> (waste + acceptable at ~74.8 s, target + inefficient at ~75.7 s with very different fill times). So the 4-class
> score below partly measures "which setting was running", not part quality. The production-relevant number is
> **waste vs acceptable inside the same regime: ROC-AUC 0.93 on clean rows** (injection pressure peak alone 0.86).
> See `PHASE4_SUMMARY.md` and `results/tables/phase4_polenta_waste_vs_acceptable.csv`.

## 1. Four quality classes (within-class chunks)

| Features / rows | Majority | Logistic regression | Random forest | Gradient boosting |
|---|---|---|---|---|
| all 13, all rows | 0.25 | 0.86 | **0.94** | 0.94 |
| all 13, clean rows | 0.25 | 0.85 | **0.90** | 0.88 |
| basic 6, all rows | 0.25 | 0.79 | **0.91** | 0.91 |
| basic 6, clean rows | 0.25 | 0.73 | **0.83** | 0.84 |

(balanced accuracy; accuracy is almost the same here, e.g. RF all 13 = 0.942; std over seeds ≤ 0.003)

- **Every mistake is between neighbouring classes** (waste↔acceptable, target↔inefficient), never two classes apart (confusion figure).
  The model has learned the ordered quality scale, which is what you'd expect from a continuous U₀ cut into bins.
- **Random forest and gradient boosting clearly beat logistic regression here (0.94 vs 0.86)**, the opposite of SKZ. With 1451 real production rows instead of 47 settings, the flexible models have enough data.
- **Pressure/force/torque channels matter:** dropping them costs 0.03 (all rows) to 0.07 (clean rows) in balanced accuracy.
- Chunks vs random rows gives almost the same *average* (0.94 vs 0.94), but the folds vary more under chunks: the worst fold is 0.85 vs 0.93.

## 2. Waste vs rest (within-class chunks)

| Features / rows | Model | ROC-AUC | PR-AUC (base rate) | Balanced acc., threshold tuned on train | Balanced acc. at 0.5 | Acceptable lenses called waste |
|---|---|---|---|---|---|---|
| all 13, all rows | RF | 0.989 | 0.97 (0.26) | 0.88 | **0.96** | 12 % |
| all 13, all rows | GB | 0.990 | 0.97 (0.26) | 0.86 | **0.96** | 11 % |
| all 13, all rows | LR | 0.91 | 0.74 (0.26) | 0.73 | 0.82 | 73 % |
| all 13, clean rows | RF | 0.957 | 0.66 (0.08) | 0.84 | **0.88** | 13 % |
| basic 6, all rows | RF | 0.979 | 0.95 (0.26) | 0.82 | 0.91 | 23 % |
| basic 6, clean rows | RF | 0.920 | 0.52 (0.08) | 0.78 | 0.78 | 20 % |

- **Ranking is excellent** (AUC 0.96–0.99), but **the threshold is again the weak point**, as on SKZ:
  - The tuned threshold *loses* 0.04–0.08 against the untuned 0.5. In folds 0 and 1 the inner CV picked a cut of 0.07–0.10, because the early chunks of the training data look different from the later ones.
  - Both numbers are leakage-free: 0.5 is a fixed default, not tuned on the test rows. The lesson is that a threshold tuned on a small, drifting history can be worse than a sensible default.
- **False alarms come only from "acceptable" lenses** (11–13 %), and most of them from its first chunk in file order. Target and inefficient lenses are never called waste. Lenses just above the U₀ = 0.40 border look like waste, so this is a physical border case, not a model bug.

## 3. The suspicious (long-decimal) rows
- With them removed, waste recall (4-class, RF) drops from **0.95 to 0.80**, and the waste PR-AUC drops from 0.97 to 0.66 (partly because the base rate falls from 26 % to 8 %).
- So the long-decimal waste rows are **easier** to recognise than the real-looking ones. That fits interpolated or synthetic points that sit between existing waste rows.
- **My reading:** the "all rows" numbers are optimistic for waste. The honest production estimate for waste detection is the **clean-rows** one: ROC-AUC ≈ 0.96 and balanced accuracy ≈ 0.84–0.88 with all 13 signals.
  This can't be confirmed without the paper's data-preparation section (MDPI blocked automated reading). If you can download the PDF, I can check it.

## 4. Comparison with your earlier quick runs
| Earlier (untuned, scheme unclear) | Now (within-class chunks) |
|---|---|
| all 13, 4-class accuracy 0.905 | RF 0.942 (all rows), 0.928 (clean rows) |
| 6 simple features 0.861 | basic 6: RF 0.912 (all rows), 0.893 (clean rows) |
| waste-vs-rest accuracy 0.942 | RF 0.956 at 0.5; 0.835 with train-tuned threshold; 0.939 clean rows at 0.5 (majority 0.917) |

## What Phase 3 adds to the project
1. **Machine data predicts optical quality in real production** (4-class balanced accuracy ≈ 0.90–0.94, waste ranking AUC ≈ 0.96–0.99), and errors stay between neighbouring classes.
2. **The threshold problem is general,** not an SKZ artefact: rankings transfer across production periods, cut-offs don't. Any product needs a calibration step.
3. **Hardware hint for Phase 4:** pressure/force/torque channels add measurable value (0.03–0.07 balanced accuracy).
4. **Data-quality warning:** part of the waste class looks synthetic, and the clean-row numbers should be the ones quoted.

## Limits
- There is no date column; the chunk order is *assumed* chronological from the row order inside each class.
- The labels come from binning one continuous measurement (U₀), so the border cases (waste/acceptable) will always be noisy.
- "Melt temperature" (82–155 °C) is implausibly low for a real melt temperature; physical interpretation in Phase 4 must be careful.
- One machine, one product, 5 days.
