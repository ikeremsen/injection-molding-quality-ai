# Phase 4 summary: which signals matter, and what our logger must record

Scripts: `phase4_skz.py` (≈15 min), `phase4_polenta.py` (≈12 min), `phase4_polenta_waste_vs_acceptable.py`,
`phase4_logger_check.py`, `phase4_report.py`. Shared code: `src/signals.py` (feature → channel map).
Tables: `results/tables/phase4_*.csv`, `phase4_tables.md`, **`phase4_logger_signal_list.csv`** (the deliverable).
Figures: `phase4_signal_selection_heatmap.png`, `phase4_selection_curves.png`, `phase4_top_shap_features.png`.

## Method (in plain words)
- A logger records **channels** (physical signals), not columns. Every feature was mapped to one of 11 channels
  (screw position, melt pressure, cavity pressure, injection speed, plastification, clamp force, phase timing,
  barrel temperature, mold/coolant temperature, TCU flow/power, setpoints). The 13 values of Dataset 2 (Polenta) use the same names.
- Three views, all on **held-out folds** (grouped CV for the Main dataset (SKZ), within-class chunks for Dataset 2 (Polenta)):
  1. **Channel permutation:** shuffle one channel and measure the score drop. On the Main dataset (SKZ) the shuffling stays inside each day, so "telling days apart" cannot count as importance.
  2. **SHAP:** which individual features move the predictions most.
  3. **Nested forward selection:** inside each training fold, add the channel that helps most (inner CV), then score on the untouched test fold. This is the honest answer to "how few channels do we need?". Permutation alone underrates channels that share information (shuffle one, the others cover for it).
- Weight is scored by **offset-corrected MAE** (each test day's mean error removed). This assumes the daily recalibration that Phase 2 showed is needed anyway.

## Fixes made during Phase 4
- **Dose alignment:** dosing in cycle n feeds shot n+1, so weight now uses the previous cycle's dose. The first logged part of each group has no logged previous cycle. Leaving it empty raised the weight error from 0.48 to 0.64 g, so it is filled with its own cycle's dose (same setting, steady state). A real logger records every cycle and would not need this.
- **The timers of Dataset 2 (Polenta) mark the production regime:** waste + acceptable were made at a cycle time of ~74.8 s, target + inefficient at ~75.7 s (with very different fill times). The "4-class" model partly recognises *which setting was running*. The honest production test is **waste vs acceptable**, the same regime (table below).

## Results

**How many channels are enough?** (nested selection, outer folds; figure `phase4_selection_curves.png`)

| Target | 1 channel | 2 channels | 3 channels | All channels | Most-picked channels |
|---|---|---|---|---|---|
| (SKZ) NOK (AUC) | 0.84 | 0.90 | 0.93 | 0.95 | melt pressure, mold/coolant temperature, cavity pressure, screw position |
| (SKZ) sink marks (AUC) | 0.92 | 0.96 | 0.98 | 1.00 | cavity pressure, injection speed, mold temperature, screw position |
| (SKZ) underfilled (AUC) | **0.965** | 0.965 | 0.95 | 0.91 | cavity pressure (first pick 15/15), clamp force |
| (SKZ) part weight (MAE, g) | 0.93 | **0.48** | 0.46 | 0.47 | screw position (15/15), clamp force (13/15), plastification |
| (Polenta) waste, clean rows (AUC) | 0.88 | 0.92 | 0.94 | 0.955 | melt pressure (first pick 15/15), plastification, screw position |

**Candidate loggers on the Main dataset (SKZ)** (grouped CV; mildly optimistic because the sets were proposed after seeing the selection):

| Configuration | NOK AUC | Sink marks AUC | Underfilled AUC | Weight MAE |
|---|---|---|---|---|
| Setpoints only (no logger) | 0.91 | 0.97 | 0.92 | 1.59 g |
| A. Cheap logger*, cycle summaries only | **0.96** | 0.997 | 0.89 | 0.78 g |
| B. Cheap logger*, with curves | 0.94 | 1.00 | 0.90 | 0.51 g |
| C. B + cavity pressure sensor | 0.955 | 1.00 | 0.91 | **0.47 g** |
| D. Everything measured | 0.95 | 1.00 | 0.905 | 0.47 g |
| Cavity pressure alone | — | — | **0.96** | — |
| Melt pressure + clamp force, summaries | — | — | 0.94 | — |

\*Cheap logger = screw position, melt pressure, clamp force, mold/coolant temperature, plastification, injection speed (all standard machine or TCU signals).

**Dataset 2 (Polenta): waste vs acceptable** (same machine regime, the hard and honest case):

| Features | Clean rows AUC | All rows AUC |
|---|---|---|
| all 13 | 0.93 | 0.98 |
| without the timers | 0.92 | 0.97 |
| injection pressure peak only | 0.86 | 0.95 |
| timers only | 0.66 | 0.81 |

## Physics interpretation
- **Melt (hydraulic) pressure, especially in the holding phase**, is the strongest general signal. The holding-pressure level and its slope decide how much the shrinking melt is topped up, and too little packing gives **sink marks**. In Polenta the injection pressure peak reflects flow resistance (melt viscosity, fill behaviour), and that separates waste from acceptable lenses.
- **Screw position** measures the volume that went in (injection stroke, dose) and whether packing still acts (cushion). That makes it the main **weight** signal: shot mass from the screw stroke and the pVT model gives r = 0.98.
- **Cavity pressure** is the most direct measure of filling: an **underfilled** part never builds up full cavity pressure, which is why it is always the first pick for underfill. It needs an in-mold sensor, though.
- **Clamp force** is the cheap stand-in for cavity pressure. The melt pressing on the projected part area pushes the mold halves apart ("mold breathing"), and the clamp force curve sees it. That is why it is picked for weight (87 %) and underfill (80 %).
- **Mold / coolant temperature** controls how fast the skin freezes. The coolant supply-return temperature difference (heat taken out of the mold) is the top SHAP feature for NOK: a hot mold with low holding pressure makes sink marks.
- **Injection speed** matters for sink marks (shear heating, pressure transmitted to the far end of the part). **Plastification** (rpm, torque) matters for weight (dose amount) and Polenta waste (melt state).
- **Barrel temperatures, timers, TCU pump/flow values, shop floor air:** little or no information about the part. The timers and day-constant values mostly identify the setting or the day, which is a risk, not a help.
- **Curves vs summaries:** defects are predicted as well from simple cycle summaries (peak, cushion, holding level). **Weight needs the curves** (0.78 → 0.51 g), mainly screw position, rpm and clamp force over the cycle.

## Decision: what our logger must record
See `results/tables/phase4_logger_signal_list.csv` for the full table with evidence.

| Priority | Signals | How |
|---|---|---|
| **MUST** | screw position, melt/hydraulic pressure, clamp force | full curves ≥100 Hz (the Main dataset (SKZ) used 200 Hz) + cycle values (cushion, injection/dosing stroke, peaks, holding level and slope) |
| **MUST** | mold temperature + coolant supply and return temperature per mold half | per cycle (1 Hz is enough) |
| **MUST (free)** | all setpoints and every setpoint change, with timestamp | from the controller (OPC-UA / EUROMAP 77) |
| SHOULD | injection speed, plastification rpm + torque | curves ≥100 Hz, cheap because they come from the same machine interface |
| OPTIONAL per mold | cavity pressure sensor | only where short shots are the main risk (+0.02 AUC on underfill here, nothing on sink marks or NOK) |
| OPTIONAL | barrel/nozzle zone temperatures, phase timers | per cycle, as context and for drift / setting-change detection; not as model inputs |
| NOT NEEDED | TCU pump/flow/heater power, shop floor air temperature/humidity | — |

**Not signals, but just as important for our own dataset** (lessons from Phases 1–4):
1. **Label every part, not every setting.** The labels of the Main dataset (SKZ) are constant per setting, so it cannot show part-to-part detection. Record defect labels and **part weight per part**, linked to the machine cycle ID.
2. **Record stable production, not only a DoE.** Most real defects are the occasional bad part at a fixed setting; we have no data of that kind yet.
3. **Log every cycle, including warm-up after setting changes.** Gaps break the shot alignment (the dose problem above) and hide transients.
4. **Timestamps, cycle counter, material + lot, mold ID, operator events** (material change, cleaning, stops). These are needed to group, validate and detect drift. Keep them out of the model inputs.
5. **A few reference parts per day/material** (weighed and inspected) for the recalibration step that every phase has shown is necessary.

## Limits
- Main dataset (SKZ): 47 settings on one machine and mold. Underfill rests on 5 groups, so its channel ranking is the weakest.
- Dataset 2 (Polenta): only 13 cycle values (no curves), no timestamps, possibly synthetic waste rows (clean rows used as the main case).
- Sampling rates below 200 Hz were not tested; "≥100 Hz" is a recommendation, not a measured result.
- Importance means "useful for prediction in this data", not proven cause. The physics explanations are consistent with the data, but the experiments were not designed to prove them.
