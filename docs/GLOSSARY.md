# Glossary

Short, plain explanations of the words used in this repository. Words in *italics* inside an entry have their own
entry. Back to the [README](../README.md).

**Molding process:** [cycle](#cycle) · [shot](#shot) · [setting point](#setting-point) · [setpoint](#setpoint) ·
[OK part / bad part](#ok-part--bad-part) · [screw position](#screw-position) · [dosing](#dosing) ·
[cushion](#cushion) · [switchover](#switchover) · [melt pressure](#melt-pressure) ·
[holding pressure](#holding-pressure) · [cavity pressure](#cavity-pressure) · [clamp force](#clamp-force) ·
[toggle vs hydraulic clamp](#toggle-vs-hydraulic-clamp) · [mold temperature and coolant](#mold-temperature-and-coolant) ·
[temperature-control unit](#temperature-control-unit) · [short shot](#short-shot) · [sink marks](#sink-marks) ·
[warpage](#warpage) · [sprue and gate](#sprue-and-gate) · [PP](#pp) · [ABS and recycled ABS](#abs-and-recycled-abs) ·
[bar](#bar) · [Hz](#hz)

**Data and machine learning:** [signal](#signal) · [curve vs per-cycle value](#curve-vs-per-cycle-value) ·
[feature](#feature) · [label](#label) · [model](#model) · [training vs testing](#training-vs-testing) ·
[unseen experiment, unseen day](#unseen-experiment-unseen-day) · [leakage ("cheating")](#leakage-cheating) ·
[threshold](#threshold) · [recalibration](#recalibration) · [AUC](#auc) · [balanced accuracy](#balanced-accuracy) ·
[shuffling check](#shuffling-check) · [correlation r](#correlation-r) · [error in grams](#error-in-grams) ·
[logistic regression and random forest](#logistic-regression-and-random-forest) ·
[anomaly detection](#anomaly-detection) · [pVT model](#pvt-model) · [MES](#mes) · [OPC-UA](#opc-ua) ·
[EUROMAP 63 and 77](#euromap-63-and-77)

---

## Molding process

### Cycle
One complete run of the machine that makes one shot: close the mold, inject, hold, cool, open, eject. A cycle
usually takes from a few seconds to about a minute (about 40–60 s in the Main dataset (SKZ)).

### Shot
The amount of molten plastic pushed into the mold in one *cycle*, and the part(s) that come out of it.

### Setting point
One combination of machine settings (for example a certain melt temperature, mold temperature, injection speed and
*holding pressure*). In the Main dataset (SKZ), each setting point is one experiment of about 12 parts made in a row.
Dataset 3 (León) calls a setting point a "batch".

### Setpoint
The target value of **one** machine setting, for example "holding pressure = 300 *bar*". A *setting point* is a
combination of setpoints.

### OK part / bad part
An OK part meets the quality requirements; a bad part does not. Some files use "NOK" ("not OK") for a bad part.
In the Main dataset (SKZ) a part is bad if it has *sink marks* or is a *short shot*.

### Screw position
Where the screw inside the barrel is at each moment. It moves forward to push plastic into the mold and back while
*dosing*, so it shows how much plastic went in.

### Dosing
The step where the screw rotates and moves back, melting fresh plastic and collecting it in front of the screw for the
next *shot*. Also called plastification.

### Cushion
The small amount of molten plastic left in front of the screw at the end of the *shot*. If the cushion is zero, the
screw cannot push any more plastic to make up for shrinkage.

### Switchover
The moment the machine stops pushing the screw forward at a set speed (filling) and starts holding a set pressure
(*holding pressure*). It is often set as a screw position or volume.

### Melt pressure
The pressure of the molten plastic in front of the screw, measured or calculated by the machine. In this repository
"melt / hydraulic pressure" means this machine-side pressure, as opposed to *cavity pressure*.

### Holding pressure
The pressure applied after the mold is filled, while the part cools and shrinks. It pushes in a little more plastic
to make up for the shrinkage; too little holding pressure causes *sink marks*.

### Cavity pressure
The pressure inside the mold cavity itself, measured by a sensor built into the mold. It shows directly how the part
fills and is packed, but the sensor is expensive and specific to one mold.

### Clamp force
The force that keeps the two mold halves closed against the pressure of the plastic inside. When the cavity is
packed with plastic, the mold is pushed apart slightly and the measured clamp force changes.

### Toggle vs hydraulic clamp
Two ways a machine closes the mold. A toggle clamp uses a lever mechanism that locks the mold shut; a hydraulic clamp
presses with a large oil-pressure cylinder. How clamp force is measured, and how sensitive it is, can differ between
the two.

### Mold temperature and coolant
The mold is kept at a set temperature by liquid (water or oil) flowing through channels in it. The temperature
difference between the coolant going in (supply) and coming out (return) shows how much heat the mold removes from
the part.

### Temperature-control unit
The device that heats or cools the coolant and pumps it through the mold. Some units can report their temperatures,
flow and power over a network.

### Short shot
A part that is not completely filled: the plastic did not reach every corner of the mold. Called "underfilled" in
the Main dataset (SKZ) files.

### Sink marks
Small dents on the surface, usually over thick sections, caused by the plastic shrinking inside while it cools.
Mostly a sign of too little *holding pressure* or too short holding time.

### Warpage
A part that bends or twists out of shape after molding because different areas shrink differently.

### Sprue and gate
The sprue is the channel (and the plastic left in it) that leads from the machine nozzle to the part; it is cut off
after molding. The gate is the narrow opening where the plastic enters the part.

### PP
Polypropylene, a common, cheap plastic used for many molded parts.

### ABS and recycled ABS
ABS (acrylonitrile butadiene styrene) is a tough plastic used for housings and covers. Recycled ABS is made partly
from used plastic; its properties can vary more from batch to batch.

### bar
A unit of pressure. 1 bar is about the air pressure around us; molding pressures are often several hundred bar.

### Hz
Hertz: how many times per second a value is recorded. 100 Hz means 100 readings every second.

---

## Data and machine learning

### Signal
A quantity the machine or a sensor measures, for example *screw position*, *melt pressure* or mold temperature.
A data logger records signals.

### Curve vs per-cycle value
A curve is a signal recorded many times during one *cycle* (for example 200 times per second), so you can see its
shape. A per-cycle value is one number per cycle, for example the peak pressure or the *cushion*.

### Feature
A number calculated from a signal and given to the *model*, for example "highest pressure during holding". This
repository mostly says "signal value" instead.

### Label
The answer we want the model to learn, recorded for each part: OK or bad, which defect, or the part weight.

### Model
A program that learns the link between machine signals and the *label* from examples, and then predicts the label
for new parts. "Machine learning" (ML) is the general name for this approach.

### Training vs testing
Training is the part of the data the model learns from. Testing is separate data used only to check how well it
works. A fair test uses data the model has never seen.

### Unseen experiment, unseen day
The model is tested on parts from a *setting point* (experiment) or a day that it did not learn from — like testing a
student on questions they have not seen before, not on the practice questions.

### Leakage ("cheating")
When the model gets information during training that it would not have in real use, so test results look better than
they really are. Example: if 11 parts of an experiment are in training and the 12th in testing, the model only has to
recognise the experiment.

### Threshold
The cut-off that turns the model's score (between 0 and 1) into a decision: above it, the part is called bad; below
it, OK.

### Recalibration
Adjusting the *threshold* (or a weight offset) for a new day, material or start-up, using a few freshly inspected
parts, because the model's scores can shift even when its ranking is still right.

### AUC
"Area under the curve": a score from 0.5 to 1.0 that says how well the model ranks bad parts above OK parts.
0.5 = no better than guessing, 1.0 = every bad part ranked above every OK part. It does not depend on the
*threshold*.

### Balanced accuracy
The average of two shares: the share of bad parts that are caught and the share of OK parts that are passed.
0.5 = guessing, 1.0 = perfect. It does depend on the *threshold*.

### Shuffling check
The labels are mixed up at random and the model is trained again. Whatever score it still reaches comes from luck
alone, which shows how high a score must be before it means something.

### Correlation r
A number from −1 to +1 that shows how closely two quantities move together on a straight line. r = 0.98 means
almost a perfect straight-line relation.

### Error in grams
For part weight, the average difference between predicted and real weight (mean absolute error), in grams.

### Logistic regression and random forest
Two common kinds of *model*. Logistic regression draws one straight dividing line between OK and bad parts; a random
forest combines many simple yes/no decision trees and can follow more complicated patterns.

### Anomaly detection
A model that learns only what normal cycles look like and flags cycles that look different. Useful when there are
too few bad parts to learn from directly.

### pVT model
A physical model of how the volume of a plastic changes with pressure (p) and temperature (T). Together with the
*screw position* it gives the mass of plastic injected in a *shot*.

### MES
Manufacturing Execution System: the factory software that collects production data (counts, times, some machine
values) from machines.

### OPC-UA
An open communication standard that lets a computer read data from industrial machines over a network.

### EUROMAP 63 and 77
Standards for how injection molding machines share data with other systems. EUROMAP 63 is an older, file-based
interface; EUROMAP 77 is a newer one based on *OPC-UA*.
