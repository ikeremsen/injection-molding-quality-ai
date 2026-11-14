# Phase 0 + Phase 1 summary (setup and data audit, no modeling)

Produced by `scripts/audit_skz.py` and `scripts/audit_polenta_leon.py`.
Raw outputs: `audit_skz.txt`, `audit_polenta_leon.txt`, `skz_columns.csv`,
`skz_labels_by_group.csv`, `skz_settings_by_group.csv`.

## Phase 0: setup
- Python 3.13.9 venv at `%USERPROFILE%\.venvs\injection-molding-ai` (kept **outside OneDrive** so that
  OneDrive does not sync ~10k package files). `requirements.txt` holds the packages, `requirements-lock.txt` the exact versions.
- Layout: `src/` (shared code: `loaders.py`), `scripts/` (one script per step), `results/{audit,figures,tables}`.
- Loaders: `load_skz(with_arrays=True/False)`, `load_leon()`, `load_polenta()` (semicolon CSV, row order kept).

---

## A) Main dataset (SKZ): matches and mismatches against the brief

**Matches:** 564 rows, 364 columns, 143 array columns, 47 groups (17 + 17 + 13), 3 days / materials as described,
NOK = SinkMarks OR Underfilled (0 exceptions), 155 NOK, day rates 46.6 / 23.5 / 7.7 %, no E77 columns,
DXP arrays at 200 Hz (0.005 s) confirmed from array length / cycle time. Every group was produced in one
continuous block of time.

**Mismatches and surprises**

1. **Labels are constant within every group.** All 47 groups are either 100 % OK or 100 % NOK (34 OK, 13 NOK).
   The same holds for SinkMarks (12 groups) and Underfilled (5 groups). The labels describe the *setting*, not a
   part-to-part event. So the effective sample size for classification is **47 groups (13 positive)**, not 564 parts.
   A model can only learn "which settings give bad parts", and AUC per part is really AUC over 47 settings repeated ~12 times.
2. **Label counts per day make some validations nearly empty:**
   - March 2022 has only **1 NOK group** (A20, underfilled) and **0 sink-mark groups**.
     Leave-one-day-out with March as the test day rests on one group, and for SinkMarks it is undefined (no positives).
   - Underfilled exists only on Dec-7 PP (4 groups) and March (1 group). ABS has none, so train-PP/test-ABS on Underfilled is undefined.
3. **Two prefixes are not in the brief: DOS (dosing, 21 array cols) and DRY (dryer, 31 array cols).** 28 of them are
   constant. DOS arrays exist only in Dec, DRY_ac* only in March. Several are counters / running totals
   (`DOS_acBatchCounter`, `*_TotalWeight`), which encode run order. They mostly identify the day.
4. **SET_ columns contain recording errors** (the measured actual values agree with the PDF, the SET_ values do not):
   | Day / exp. | Column | SET_ value | PDF value | Measured actual |
   |---|---|---|---|---|
   | 2021-12-07 A16 | SET_HoldingPressure1 | 200 | 60 | 59.8 (DXP_HoldingPressure1) |
   | 2021-12-15 A02 | SET_HoldingPressure1 | 350 | 200 | 200.1 |
   | 2021-12-15 A07, A08 | SET_InjectionSpeed | 80 | 20 | 15.1 (QUA_InjectionSpeed1) |
   | 2021-12-15 all | SET_TransferVolume | copy of cooling time (18/22/23) | 20.25–25.5 | matches PDF (QUA_TransferStroke) |
   | 2022-03-17 A32 | SET_HoldingTime1/2 | 0 / 0 | 18 / 7 (two-stage) | 7.1 / 17.9 |
   The A16 error matters: A16 is a NOK group, and with the wrong SET value it looks identical to the OK group A15.
5. **SET_BackPressure is missing on both December days** (not varied, value not recorded).
6. **Group sizes:** 2021-12-07 A15 has 13 parts and A16 has 11. The measured holding pressure confirms the assignment, so this is real, not a mislabel.
7. **Columns missing on whole days** (so they cannot be used for cross-day tests without imputing a whole day):
   QUA_InjectionSpeed1, QUA_HoldingPressureMax, QUA_MeltPressure2Max, QUA_PlastificationSpeed, TCE_ActualFlowRate(Mean)
   are missing in March. CALC_* are missing for ABS (by design: Tait model only for PP). `CV_Width` exists only in March.
8. **Day / time proxies among the *allowed* inputs:**
   - ENV is constant inside every group (the sensor value is held per experiment). Humidity is 26–32 % / 30–40 % / **70 % fixed** on the three days.
     On its own, humidity separates NOK with AUC 0.76, but that comes from telling the days apart, not from physics.
   - TCE/TCN pump power and pump-speed means are constant within each day: they only identify the day.
   - QUA_OilTemperature rises steadily through each day (machine warm-up), so it tracks run order.
   These columns can make a model look better under GroupKFold, because days differ in NOK rate.
9. **Regression targets are dominated by the material.** SCA_PartWeight: ABS ~111 g vs PP ~91–96 g; the overall std is 9.1 g,
   the within-group std is 0.02 g. A model that only knows the material already gets a high R², so R² must be reported **within each day**.
   CV_Warpage has an outlier (7.1 mm in March, the next highest is ~2.3 mm).
10. Strong single features (screening on all rows, no CV): CALC_InjectionStroke / DosingStroke 0.94, QUA_CushionVolume 0.91,
    cavity-pressure integrals 0.86, holding pressure 0.83–0.85. These make physical sense (shot volume, packing), but CALC is missing for ABS.
11. A32 (March) repeats the settings of A01 (Dec-7). Both are OK. This pair is a natural check for drift between days.

---

## B) Dataset 3 (León): real columns and what the paper does

**Columns:** `batch, label_viability, Tinj, tinj, Pinj, Ph, Bp, th, P1_weight … P5_weight, weight_avg, energy, cycle_time`.

**From the paper (Pérez-González et al., IJAMT 2026) and the Zenodo record:**
- Machine: Mateu & Solé Minimat 60, PP, the part is two 55 mm discs. 8 shots per batch: 3 thrown away (transient), 5 weighed.
- Design: batches 1–768 are a full factorial (4·4·4·3·2·2 levels; confirmed: exactly the first 768 rows) and batches 769–1000 are maximin space-filling.
- Labels (Zenodo): **G** = part produced (may still have defects), **Y** = produced but with clear **short shots**, **R** = no part / injection incomplete or erratic.
- Objectives: (1) energy (surrogate), (2) |predicted weight − **5.645 g**| (surrogate), (3) cycle time = **15.66 + tinj + th** (a formula, no model needed; confirmed exact in the CSV).
- Constraint: within-batch variability = max |part weight − batch median| ≤ **0.073 g** (the mean of that variability over all batches), predicted by a third surrogate. Range limits are those in Table 1 (also 20.16 ≤ cycle time ≤ 23.66 s).
- Validation in the paper: random 80/20 split of configurations, then 7 Pareto points re-run on the machine. Weight was predicted well (MAE ~0.04 g),
  energy less well (MAPE ~14 %), and variability poorly (MAPE 55–130 %). 3 of the 7 re-run points broke the variability limit.

**Mismatches and things to watch**
- Paper: 257 non-injectable configs / 743 injectable. CSV: **260 R** / 740 G+Y. Batch 424 is labelled G but has no weights.
- **Energy units / scale do not match the paper.** The paper's Table 8 and its 10 Wh penalty imply about 6 Wh per configuration. The CSV energy has a median of **41 Wh**
  ("per batch" according to Zenodo) and a maximum of 268 Wh. So the CSV is probably the batch total, maybe including the 3 discarded shots. I will model the CSV value as it is, and **not** compare it with the paper's energy numbers.
- Energy is missing in 223 rows, including 10 % of G rows. 121 R rows still have small energy values (median 0.3 Wh).
- Single-part weight outliers, probably typos or a double part: batch 408 P3 = 8.83 g, and 6 more parts above 6.6 g. These blow up the variability metric for those batches.
- My recomputed variability (mean 0.073, median 0.048, p95 0.137 g) matches the paper exactly. Good.
- Y batches are on average 0.4 g under target (short shots), which is consistent with the label.
- There is a companion Zenodo record with surface-defect images for the same batches (10.5281/zenodo.20322729). It is not used now, but noted for later.

---

## C) Dataset 2 (Polenta): real data versus the readme

- 1451 rows, 13 features + `quality`, no missing values, no duplicates. Class counts: **370 / 406 / 310 / 365**
  (the readme says 360 for class 4, which gives 1446 rows, not 1451).
- **The rows are sorted by class, not by time:** 370 × class 1, then 406 × class 2, 310 × class 3, 365 × class 4 (only 3 label changes in the whole file).
  So **contiguous block CV on this order is not valid**: each block would hold mostly one class, and the model would be tested on a class it barely saw in training.
  The preliminary "block CV 0.905" therefore cannot have used purely contiguous blocks of this order.
- **Possible synthetic or processed rows:** values with more than 4 decimals (e.g. 919.4097910069526) cannot come from the MES, which logs 1–3 decimals.
  **75 % of waste rows** contain at least one such value, compared with 5 % / 9 % / 0.3 % in the other classes. This looks like oversampling or interpolation of the waste class,
  or a different processing step. It could let a model separate "waste" by artefacts. I could not confirm this from the paper (the MDPI site blocks automated reading).
- Inside each class the slow machine states (mold temperature, cycle time) jump at a few places, which may be day boundaries. So the order inside a class may be chronological, but this is not proven.
- "Melt temperature" ranges from 82 to 155 °C, too low for a real melt temperature of a lens material. It is probably a sensor at another location; noted for interpretation.
