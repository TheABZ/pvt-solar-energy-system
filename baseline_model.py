"""Three-node PV/T solar collector with verification and flow optimization.

Install: pip install numpy scipy matplotlib
Run:     python pvt_model_commented.py
All input parameters are illustrative, not experimentally calibrated.
"""

# Paths allow all outputs to be saved beside this script.
from pathlib import Path
import json

# Arrays, ODE integration, quadrature, and plotting are separate tools.
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib
matplotlib.use("Agg")  # Write figures without requiring a desktop display.
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent

# Optical and electrical properties; efficiencies are fractions.
A = 1.6                 # Collector aperture area, m^2.
ALPHA = 0.90            # Absorbed fraction of incident radiation.
ETA_REF = 0.20          # Electrical efficiency at reference temperature.
BETA = 0.004            # Relative efficiency loss per kelvin.
TA = 25.0               # Constant ambient air temperature, deg C.
TIN = 25.0              # Constant inlet water temperature, deg C.
TREF = 25.0             # Electrical reference temperature, deg C.

# Lumped thermal capacities; the cell includes its associated laminate.
CC = 12000.0            # Cell-laminate capacity, J/K.
CB = 6000.0             # Backplate capacity, J/K.
CP = 4180.0             # Water specific heat, J/(kg K).
CW = 1.0 * CP           # Capacity of 1 kg of channel water, J/K.

# Conductances already include effective heat-transfer area.
HCA = 19.2              # Cell to environment, W/K.
HCB = 120.0             # Cell to backplate, W/K.
HBA = 3.2               # Backplate to environment, W/K.
HBW = 160.0             # Backplate to water, W/K.
END = 12.0 * 3600.0     # Twelve-hour simulation duration, seconds.


def irradiance(t):
    """Synthetic irradiance, W/m^2, for the simulated daytime interval."""
    # Half a sine wave gives zero at sunrise and sunset and 900 at noon.
    return 900.0 * np.maximum(0.0, np.sin(np.pi * np.asarray(t) / END))


def efficiency(tc, beta=BETA):
    """Temperature-dependent gross electrical efficiency, dimensionless."""
    return ETA_REF * (1.0 - beta * (tc - TREF))


def rhs(t, y, flow, beta, sun):
    """Return three temperature derivatives in deg C per second."""
    tc, tb, tw = y
    g = sun * irradiance(t)  # sun=0 enables the darkness verification.
    pel = efficiency(tc, beta) * A * g

    # Positive flow activates water coupling. Zero selects a DRY panel.
    # A stagnant-water experiment would retain HBW at zero flow instead.
    qbw = HBW * (tb - tw) if flow > 0.0 else 0.0
    qcb = HCB * (tc - tb)           # Internal cell-to-backplate heat.
    qfront = HCA * (tc - TA)        # Front environmental heat loss.
    qback = HBA * (tb - TA)         # Rear environmental heat loss.
    useful = flow * CP * (tw - TIN) # Net outlet enthalpy transport.

    # Subtract electricity before assigning absorbed radiation to heat.
    dtc = (ALPHA * A * g - pel - qfront - qcb) / CC
    # The same qcb enters the backplate that leaves the cell.
    dtb = (qcb - qback - qbw) / CB
    # Water stores the difference between incoming heat and outlet heat.
    dtw = (qbw - useful) / CW
    return [dtc, dtb, dtw]


def simulate(flow, rtol=1e-7, step=60.0, dt=30.0, beta=BETA, sun=1.0):
    """Solve one case and return time histories and energy statistics."""
    # Reporting points are distinct from the solver's internal time steps.
    t = np.linspace(0.0, END, int(END / dt) + 1)
    sol = solve_ivp(
        rhs, (0.0, END), [TA, TA, TIN],
        args=(flow, beta, sun), t_eval=t, method="LSODA",
        rtol=rtol, atol=rtol / 10.0, max_step=step
    )
    if not sol.success:
        raise RuntimeError(sol.message)

    tc, tb, tw = sol.y
    eta = efficiency(tc, beta)
    # An invalid efficiency requires revisiting the model, not clipping.
    if np.any(eta <= 0.0) or np.any(eta >= ALPHA):
        raise ValueError("Efficiency outside the model operating range")
    pel = eta * A * sun * irradiance(t)
    qu = flow * CP * (tw - TIN)

    # The illustrative pump law assumes continuous daytime operation.
    pump = 5.0 * (flow / 0.020) ** 3 if flow > 0.0 else 0.0
    es = trapezoid(A * sun * irradiance(t), t)
    ee = trapezoid(pel, t)          # Watts times seconds gives joules.
    eth = trapezoid(qu, t)

    # Internal heat flows cancel when all three balances are added.
    net = (
        ALPHA * A * sun * irradiance(t) - pel
        - HCA * (tc - TA) - HBA * (tb - TA) - qu
    )
    initial = np.array([TA, TA, TIN])
    stored = np.dot([CC, CB, CW], sol.y[:, -1] - initial)
    residual = trapezoid(net, t) - stored
    peak = int(np.argmax(tc))

    # Convert energies to kWh and report a normalized conservation error.
    stats = {
        "flow_kg_s": float(flow),
        "peak_cell_C": float(tc[peak]),
        "peak_clock_hour": float(6.0 + t[peak] / 3600.0),
        "min_efficiency_pct": float(100.0 * eta.min()),
        "solar_kWh": float(es / 3.6e6),
        "electricity_kWh": float(ee / 3.6e6),
        "heat_kWh": float(eth / 3.6e6),
        "pump_kWh": float(pump * END / 3.6e6),
        "net_electricity_kWh": float((ee - pump * END) / 3.6e6),
        "daily_electric_efficiency_pct": 100.0 * ee / es if es else None,
        "daily_thermal_efficiency_pct": 100.0 * eth / es if es else None,
        "energy_residual_pct": (
            100.0 * abs(residual) / (ALPHA * es) if es else None
        ),
        "peak_water_C": float(tw.max()),
    }
    return t, sol.y, eta, pel, qu, stats


def main():
    # Baseline comparison: identical sunlight and environmental losses.
    dry = simulate(0.0)
    wet = simulate(0.020)
    report = {"dry": dry[-1], "cooled": wet[-1]}

    # Tighten integration controls and refine the output quadrature grid.
    fine = simulate(0.020, rtol=1e-9, step=15.0, dt=15.0)
    report["refinement"] = {
        "peak_temperature_change_C": abs(
            fine[-1]["peak_cell_C"] - wet[-1]["peak_cell_C"]
        ),
        "electricity_change_kWh": abs(
            fine[-1]["electricity_kWh"] - wet[-1]["electricity_kWh"]
        ),
    }

    # Independent physical limits test equilibrium and electrical coupling.
    dark = simulate(0.020, sun=0.0)
    no_beta_dry = simulate(0.0, beta=0.0)
    no_beta_wet = simulate(0.020, beta=0.0)
    report["checks"] = {
        "dark_max_temperature_deviation_C": float(
            np.max(abs(dark[1] - 25.0))
        ),
        "beta_zero_gross_electricity_difference_kWh": abs(
            no_beta_dry[-1]["electricity_kWh"]
            - no_beta_wet[-1]["electricity_kWh"]
        ),
    }

    # Search a finite grid; this is not an exact continuous optimum.
    flows = np.linspace(0.005, 0.060, 23)
    sweep = [simulate(float(m))[-1] for m in flows]
    best = max(sweep, key=lambda s: s["net_electricity_kWh"])
    report["best_tested_flow"] = best
    report["sweep"] = sweep
    if best["flow_kg_s"] in (float(flows[0]), float(flows[-1])):
        print("Best point is at a boundary; extend the flow interval.")
    if best["net_electricity_kWh"] <= dry[-1]["electricity_kWh"]:
        print("Dry baseline gives at least as much net electricity.")

    # Use simultaneous noon values as well as separate daily extrema.
    j = int(np.argmin(abs(wet[0] - END / 2.0)))
    report["noon"] = {
        "dry_cell_C": float(dry[1][0, j]),
        "cooled_cell_C": float(wet[1][0, j]),
        "dry_efficiency_pct": float(100.0 * dry[2][j]),
        "cooled_efficiency_pct": float(100.0 * wet[2][j]),
        "cooled_backplate_C": float(wet[1][1, j]),
        "cooled_water_C": float(wet[1][2, j]),
    }
    (OUT / "results.json").write_text(json.dumps(report, indent=2))

    # Store all three cooled-node temperatures for subsequent analysis.
    hours = 6.0 + wet[0] / 3600.0
    data = np.column_stack([
        hours, irradiance(wet[0]), dry[1][0], *wet[1],
        dry[2] * 100.0, wet[2] * 100.0, dry[3], wet[3], wet[4]
    ])
    header = (
        "hour,G_W_m2,Tc_dry_C,Tc_cooled_C,Tb_C,Tw_C,"
        "eta_dry_pct,eta_cooled_pct,P_dry_W,P_cooled_W,Q_water_W"
    )
    np.savetxt(
        OUT / "pvt_results.csv", data,
        delimiter=",", comments="", header=header
    )

    # Create the required temperature and efficiency subplots.
    fig, ax = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True, constrained_layout=True
    )
    cases = [
        (dry, "firebrick", "Without water cooling"),
        (wet, "royalblue", "Water cooling: 0.020 kg/s"),
    ]
    for result, color, label in cases:
        ax[0].plot(hours, result[1][0], color=color, label=label, lw=2)
        # Mask zero sunlight: conversion efficiency is not measured then.
        daylight_eta = np.where(
            irradiance(result[0]) > 1.0, result[2] * 100.0, np.nan
        )
        ax[1].plot(hours, daylight_eta, color=color, label=label, lw=2)
    ax[0].set(
        ylabel="Cell temperature (deg C)",
        title="Illustrative PV/T model: 12-hour daytime response"
    )
    ax[1].set(
        ylabel="Electrical efficiency (%)", xlabel="Clock hour",
        xlim=(6.0, 18.0)
    )
    for panel in ax:
        panel.legend()
        panel.grid(alpha=0.25)
    fig.savefig(OUT / "pvt_comparison.png", dpi=200)
    plt.close(fig)
    print(json.dumps({k: v for k, v in report.items() if k != "sweep"},
                     indent=2))


if __name__ == "__main__":
    main()
