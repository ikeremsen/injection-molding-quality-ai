# Phase 2 summary: quality prediction on the Main dataset (SKZ)

Scripts: `scripts/phase2_skz_classification.py` (≈33 min on 12 cores), `scripts/phase2_skz_regression.py` (≈2 min),
`scripts/phase2_report.py` (figures + `results/tables/phase2_tables.md`). Raw run tables: `results/tables/phase2_*_runs.csv`.

## Setup in one paragraph
- **Feature ladder:** L1 settings (corrected from the PDF) → L2 machine cycle values (QUA + DXP scalars + CALC; oil temperature removed because it is a warm-up clock)
  → L3 + temperature-control means and ENV (as in the brief, *contains day proxies*) → L3-clean (TC means without the day-constant ones, no ENV)
  → L4 + 111 curve features. The curve features are max / mean / area / slope / end value per phase (injection, holding, plastification), cut by the machine's own trigger signals, for melt pressure, cavity pressure, screw position, speed and rpm, plastification torque and clamp force.
- **Models:** majority class, single-feature threshold (a depth-1 tree that picks its feature and threshold on training data), logistic regression, random forest, gradient boosting. Fixed, modest hyperparameters; no search.
- **Validation:** stratified grouped 5-fold CV (group = day + experiment) × 5 seeds, and leave-one-day-out × 3 seeds. "Train PP → test ABS" is *the same split* as leaving out the ABS day, so it is reported once.
  Imputer and scaler are fitted inside each training fold. The threshold is tuned by an inner grouped 3-fold CV on the training rows only.

## Sanity checks done before trusting anything
| Check | Result |
|---|---|
| Labels shuffled between groups, logistic regression L3-clean | AUC mean 0.51, **95th percentile 0.64**, max 0.73 (20 shuffles). With 47 groups, AUC ≈ 0.65 can happen by luck. |
| Effect of the SET_ corrections (settings only, grouped CV) | RF 0.72 → 0.88, LR 0.78 → 0.91, GB 0.80 → 0.93. This explains your earlier 0.68. |
| Majority baseline shows AUC 0.44, not 0.50 | Pooling out-of-fold predictions mixes folds with different training NOK rates. This makes pooled AUC slightly *pessimistic*, not optimistic. |
| GB on the ABS test day = exactly 0.50 | GB gives every ABS part the same probability (≈0): no information, not "random". |

## 1. LBL_NOK (13 NOK groups of 47)
Grouped 5-fold CV, L3-clean (mean over 5 seeds):

| Model | ROC-AUC | PR-AUC (base rate 0.27) | Balanced acc. (train-tuned threshold) | FP / FN parts |
|---|---|---|---|---|
| majority | 0.44 | 0.25 | 0.50 | 0 / 155 |
| single-feature threshold (holding pressure) | 0.71 ± 0.06 | 0.59 | 0.58 | 258 / 33 |
| **logistic regression** | **0.96 ± 0.01** | **0.95** | **0.93** | 16 / 16 |
| random forest | 0.94 ± 0.01 | 0.89 | 0.88 | 45 / 19 |
| gradient boosting | 0.95 ± 0.02 | 0.86 | 0.87 | 50 / 23 |

Along the ladder (figure `phase2_LBL_NOK_auc_ladder.png`):
- **Settings alone already give 0.88–0.93** once corrected. Cycle values add a little (LR 0.91 → 0.96). That is expected, because the labels describe the settings.
- **The day-proxy step adds nothing** (0.95 with proxies vs 0.96 without, once oil temperature is out). The earlier 0.90 → 0.93 jump was very likely a proxy effect.
- **Curve features do not help classification** (LR 0.95, RF 0.92, GB drops to 0.87). 111 extra columns for 47 groups mostly add noise.
- Logistic regression is as good as or better than the tree models. With 47 effective samples, a simple model is the right choice.

**Across days** (logistic regression, L3-clean):

| Test day | NOK groups in test | ROC-AUC | Balanced acc. @0.5 | Balanced acc. @train-tuned threshold |
|---|---|---|---|---|
| 2021-12-07 PP | 8 | 0.98 | 0.77 | 0.80 |
| 2021-12-15 ABS (= train PP, test ABS) | 4 | 0.92 | 0.50 | 0.50 |
| 2022-03-17 PP | 1 | 0.83 | 0.49 | 0.46 |

- **The ranking transfers, the threshold does not.** On ABS, LR still orders the parts well, but all its probabilities are below 0.013, so every threshold learned on PP calls every ABS part "OK".
  Tree models are worse here: RF 0.59–0.68 and GB constant. A linear model extrapolates to a new material more gracefully than trees do.
- The single-feature rule scores 0.50 on every new day: a fixed holding-pressure threshold learned on one day lies outside the other day's range.
- **Practical meaning:** a deployed model will need a short **recalibration** per material / day (a handful of labelled parts at start-up to reset the threshold). That is a requirement for the product, not a tuning detail.
- The March result rests on **one** NOK group (A20), so treat 0.83 as an anecdote.

## 2. LBL_SinkMarks (12 groups) and LBL_Underfilled (5 groups)
- **Sink marks:** grouped CV LR **1.00**, RF 0.99. Leave-out ABS: LR 0.95 (L3-clean). The March test is **not computable** (no sink marks in March).
  Sink marks are almost perfectly explained by packing (holding pressure) and the related cycle values. That is physically plausible and is checked properly in Phase 4.
- **Underfilled:** grouped CV RF 0.94, LR 0.90, PR-AUC 0.80–0.86 (base rate 0.11). Cross-day numbers rest on **1 group** (March test) or on a model trained with 1 positive group (Dec-7 test).
  Values such as RF 1.00 and GB 0.00 for March show how unstable a 1-group test is. Not usable as evidence.

## 3. SCA_PartWeight regression (figure `phase2_partweight_mae_ladder.png`)
Overall R² is 0.9–0.99 almost everywhere, but that only says "the model knows PP from ABS". The honest numbers:

| Split | Best model / features | MAE | R² within day | Note |
|---|---|---|---|---|
| Grouped 5-fold CV | RF, L4 curves | **0.48 g** | **0.86** | Without curve features, R² within day ≤ 0.21 |
| Grouped 5-fold CV | Ridge, L4 curves | 0.49 g | 0.76 | |
| PP day → other PP day (March → Dec-7) | single feature `CALC_DosedWeight` | **1.03 g** | — | r = 0.98 |
| PP day → other PP day (Dec-7 → March) | single feature `CALC_DosedWeight` | **0.50 g** | — | r = 0.98; RF/GB/Ridge 2–20 g |
| Both PP days → ABS | anything | 13–23 g | ≪ 0 | Impossible: no density/material information |

- **Curve features matter for weight, not for defects.** Weight depends on how much melt went in and how well it was packed, which the screw-position and pressure curves measure directly.
- **The best cross-day weight predictor is a physics feature, not an ML model.** `CALC_DosedWeight` is the shot mass from the screw stroke and the pVT (Tait) density model. It sits about 2.8 g above the part weight (≈ the sprue, which is clipped off before weighing) with r = 0.98 on both PP days.
  The ML models rank parts reasonably inside a new day (r 0.7–0.9) but miss the absolute level by grams.
  Caveat: by the documentation, dosing in cycle *n* prepares shot *n+1*, so this feature describes the next shot. It works here because the settings are constant within a group.
- `CALC_*` exists only for PP (no Tait parameters for ABS), which is why nothing works for ABS.

## What this means for the target
1. **Defect prediction from machine data works inside the DoE** (AUC ≈ 0.95), but it is learning "which of 47 settings are bad". It is *not yet* evidence of detecting part-to-part defects under stable production. The data has no within-group label variation to test that.
2. **Transfer to a new day/material needs recalibration.** The ranking survives better than the probability scale. Linear models transfer better than trees.
3. **For the logger** (input for Phase 4): screw position (→ injection stroke / dosed volume / cushion), holding pressure, cavity pressure and melt temperature look central. Day-level context (ENV, TC pump values) added nothing.

## Open points / things to watch
- Every number from the Main dataset (SKZ) rests on 47 settings from 3 days, with extreme DoE corners. The permutation null shows AUCs up to ~0.65 by chance.
- The GB setting (depth 3, 200 iterations) is untuned on purpose. Its instability (constant predictions, std up to 6 in within-day R²) is part of the finding: avoid it at this sample size.
