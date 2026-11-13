"""Map every model feature to the physical signal ("channel") it comes from.

A data logger records channels (sensors / machine signals), not individual
features. So Phase 4 measures importance per channel: all features derived from
one channel are permuted, dropped or added together.

The same channel names are used for SKZ and Polenta, so the two datasets can be
compared in one table.
"""
import numpy as np
import pandas as pd

# name -> (what it measures, where it comes from, typical cost)
CHANNEL_INFO = {
    "setpoints":           ("machine setpoints (target values)", "machine controller (OPC-UA / EUROMAP 77)", "free if the controller can be read"),
    "screw_position":      ("screw position -> shot volume, cushion, dosing stroke", "machine (position transducer)", "standard signal"),
    "melt_pressure":       ("hydraulic / melt pressure at the screw (injection, holding, back pressure)", "machine (pressure transducer)", "standard signal"),
    "cavity_pressure":     ("pressure inside the mold cavity", "extra piezo sensor in the mold", "expensive, mold-specific"),
    "injection_speed":     ("screw speed during injection, injection power", "machine", "standard signal"),
    "plastification":      ("screw rotation, plastification torque and time", "machine (drive)", "standard signal"),
    "clamp_force":         ("clamping force", "machine (tie-bar / hydraulic)", "standard signal"),
    "phase_timing":        ("durations of injection, holding, cooling, mold movements, cycle time", "machine timers", "standard signal"),
    "barrel_temperature":  ("barrel / nozzle / flange temperatures", "machine heater zones", "standard signal"),
    "mold_temperature":    ("mold / coolant temperature", "temperature-control unit or mold thermocouple", "cheap"),
    "tcu_flow_power":      ("coolant flow, pressure, pump and heater power", "temperature-control unit (EUROMAP 82)", "needs a networked TCU"),
}


def skz_channel(col: str) -> str:
    """Channel of one SKZ feature column."""
    c = col
    if c.startswith("SET_"):
        return "setpoints"
    if c.startswith("CRV_"):
        if c.endswith("_duration"):
            return "phase_timing"
        for key, ch in [("Inj1Pos", "screw_position"), ("screw_min", "screw_position"),
                        ("Inj1Prs", "melt_pressure"), ("MldCavPrs1", "cavity_pressure"), ("cav_", "cavity_pressure"),
                        ("Inj1Vel", "injection_speed"), ("Inj1Rpm", "plastification"),
                        ("DriveSiePlst1Trq", "plastification"), ("ClpFce", "clamp_force")]:
            if key in c:
                return ch
    if c.startswith("CALC_"):
        return "screw_position"
    if c.startswith(("TCE_", "TCN_")):
        return "mold_temperature" if "Temperature" in c else "tcu_flow_power"
    if c in ("DXP_HoldingPressure1", "DXP_HoldingPressure2"):
        return "melt_pressure"
    if c in ("DXP_HoldingTime1", "DXP_HoldingTime2"):
        return "phase_timing"
    if c == "DXP_AreaCavityPressure":
        return "cavity_pressure"
    if c.startswith("QUA_"):
        if "Cavity" in c:
            return "cavity_pressure"
        if any(k in c for k in ("Cushion", "TransferStroke", "InjectionStroke")):
            return "screw_position"
        if any(k in c for k in ("Pressure",)):
            return "melt_pressure"
        if any(k in c for k in ("InjectionSpeed", "AreaPower")):
            return "injection_speed"
        if "Plastification" in c:
            return "plastification"
        if "CylinderTemperature" in c or "Flange" in c:
            return "barrel_temperature"
        if "ToolTemperature" in c:
            return "mold_temperature"
        if "Time" in c:
            return "phase_timing"
    raise KeyError(f"no channel rule for {col}")


POLENTA_CHANNELS = {
    "Melt temperature": "barrel_temperature",
    "Mold temperature": "mold_temperature",
    "time_to_fill": "phase_timing",
    "ZUx - Cycle time": "phase_timing",
    "ZDx - Plasticizing time": "plastification",
    "Ms - Torque peak value current cycle": "plastification",
    "Mm - Torque mean value current cycle": "plastification",
    "SKx - Closing force": "clamp_force",
    "SKs - Clamping force peak value": "clamp_force",
    "APSs - Specific back pressure peak value": "melt_pressure",
    "APVs - Specific injection pressure peak value": "melt_pressure",
    "CPn - Screw position at the end of hold pressure": "screw_position",
    "SVo - Shot volume": "screw_position",
}


def channel_table(cols, mapper) -> pd.DataFrame:
    """One row per feature: feature, channel, is_curve_feature."""
    return pd.DataFrame({"feature": cols, "channel": [mapper(c) for c in cols],
                         "curve_feature": [c.startswith("CRV_") for c in cols]})


def add_previous_cycle_dosing(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the dosing features to the shot they actually feed.

    Dosing (plastification) in cycle n prepares the melt for shot n+1, so for the
    part made in cycle n the relevant dose was measured in cycle n-1. We take it
    from the previous MACHINE cycle only if that cycle was logged (consecutive
    MET_MachineCycleID).

    Gap filling: the first logged part of each group follows unlogged warm-up
    cycles that already ran with the SAME setting. Its previous dose is therefore
    filled with the dose measured in its own cycle (same setting, steady state).
    A real logger records every cycle, so it would never need this fill.
    Without it, ~8 % of parts get a median filler value and the weight error
    rises from 0.48 g to 0.64 g (checked in Phase 4).
    """
    d = df.sort_values("MET_MachineCycleID")
    prev_ok = d["MET_MachineCycleID"].diff().eq(1)
    for c in ["CALC_DosedWeight", "CALC_DosingStroke"]:
        prev = d[c].shift(1).where(prev_ok).fillna(d[c])
        df.loc[d.index, c + "Prev"] = prev.values
    return df
