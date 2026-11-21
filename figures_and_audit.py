"""Regenerate the paper's simulation figures (Figs. 3-7) and audit numbers.

Run after installing the requirements:  python figures_and_audit.py
Everything is computed from baseline_model.py, so the figures and the
tabulated values come from the same equations and parameters as the paper.

Outputs, written to figures/ beside this script:
  fig3_node_temperatures.png   cell, backplate and water temperatures
  fig4_flow_sweep.png          gross and net electricity versus water flow
  fig5_temperature_surface.png cell temperature over time and flow
  fig6_heat_flows.png          heat-flow histories and noon partition
  fig7_noon_dynamics.png       node temperature gaps and rates near noon
  audit_results.json           noon rates (Table 6), daily accounting
                               (Table 8), Jacobian eigenvalues and checks
"""

from pathlib import Path
import json

import numpy as np
from scipy.integrate import trapezoid
import matplotlib
matplotlib.use("Agg")  # Save figures without a desktop display.
import matplotlib.pyplot as plt

import baseline_model as bm

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
REF_FLOW = 0.020
FLOWS = np.linspace(0.005, 0.060, 23)  # Same grid as the paper's sweep.
DPI = 300

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "savefig.bbox": "tight"})


def flows_from_run(run, flow, hba=bm.HBA):
    """Heat and electrical rates (W) along a solved trajectory."""
    t, y, eta, pel, qu, _ = run
    tc, tb, tw = y
    g = bm.irradiance(t)
    q = {
        "absorbed": bm.ALPHA * bm.A * g,
        "electrical": pel,
        "front": bm.HCA * (tc - bm.TA),
        "rear": hba * (tb - bm.TA),
        "cell_to_backplate": bm.HCB * (tc - tb),
        "backplate_to_water": bm.HBW * (tb - tw) if flow > 0 else 0 * t,
        "water": qu,
    }
    # Storage is whatever is not exported or lost: the first-law remainder.
    q["storage"] = (q["absorbed"] - q["electrical"] - q["front"]
                    - q["rear"] - q["water"])
    return q


def derivatives(run, flow):
    """Exact node temperature rates (K/s) from the model's own equations."""
    t, y = run[0], run[1]
    return np.array([bm.rhs(ti, yi, flow, bm.BETA, 1.0)
                     for ti, yi in zip(t, y.T)]).T


def jacobian(g, flow):
    """Analytic temperature Jacobian at frozen irradiance g (1/s)."""
    ae = bm.ETA_REF * bm.BETA * bm.A * g  # Electrical feedback, W/K.
    m = flow * bm.CP
    return np.array([
        [(ae - bm.HCA - bm.HCB) / bm.CC, bm.HCB / bm.CC, 0.0],
        [bm.HCB / bm.CB, -(bm.HCB + bm.HBA + bm.HBW) / bm.CB, bm.HBW / bm.CB],
        [0.0, bm.HBW / bm.CW, -(bm.HBW + m) / bm.CW],
    ])


def clock(t):
    return 6.0 + np.asarray(t) / 3600.0


def label(ax, text):
    ax.set_title(text, loc="left", fontweight="bold")


def main():
    dry = bm.simulate(0.0)
    wet = bm.simulate(REF_FLOW)
    t, y = wet[0], wet[1]
    hours = clock(t)
    noon = int(np.argmin(abs(t - bm.END / 2.0)))
    qw, qd = flows_from_run(wet, REF_FLOW), flows_from_run(dry, 0.0)

    # Fig. 3: three node temperatures in the cooled collector.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for row, name, color in zip(y, ["Cell laminate", "Backplate", "Water outlet"],
                                ["#1f3b73", "#c46a1b", "#2a8a6e"]):
        ax.plot(hours, row, color=color, lw=2, label=name)
    ax.set(xlabel="Clock hour", ylabel="Temperature (°C)", xlim=(6, 18))
    ax.legend()
    fig.savefig(OUT / "fig3_node_temperatures.png", dpi=DPI)
    plt.close(fig)

    # Fig. 4: gross and net electricity across the flow grid.
    sweep = [bm.simulate(float(m))[-1] for m in FLOWS]
    gross = np.array([s["electricity_kWh"] for s in sweep])
    net = np.array([s["net_electricity_kWh"] for s in sweep])
    best = int(np.argmax(net))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(FLOWS, gross, color="#1f3b73", lw=2, label="Gross electricity")
    ax.plot(FLOWS, net, color="#c46a1b", lw=2, label="Net electricity after pumping")
    ax.axhline(dry[-1]["electricity_kWh"], color="0.4", ls="--", lw=1.2,
               label="Dry reference")
    ax.plot(FLOWS[best], net[best], "ko", ms=6)
    ax.annotate(f"Best tested flow\n{FLOWS[best]:.4f} kg/s",
                (FLOWS[best], net[best]), xytext=(0.016, 1.45),
                textcoords="data", arrowprops=dict(arrowstyle="-", color="0.2"))
    ax.set(xlabel="Water mass flow (kg/s)", ylabel="Daily electrical energy (kWh)")
    ax.legend(loc="lower left")
    fig.savefig(OUT / "fig4_flow_sweep.png", dpi=DPI)
    plt.close(fig)

    # Fig. 5: cell temperature surface; each flow is its own transient run.
    tc_grid = np.array([bm.simulate(float(m))[1][0] for m in FLOWS])
    H, M = np.meshgrid(hours, FLOWS)
    fig = plt.figure(figsize=(7.5, 5.5))
    ax = fig.add_subplot(projection="3d")
    surf = ax.plot_surface(H, M, tc_grid, cmap="viridis", linewidth=0,
                           antialiased=True, rcount=len(FLOWS), ccount=200)
    ax.set(xlabel="Clock hour", ylabel="Mass flow (kg/s)", zlabel="Cell temperature (°C)")
    ax.view_init(elev=25, azim=-130)
    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, label="Cell temperature (°C)")
    fig.savefig(OUT / "fig5_temperature_surface.png", dpi=DPI)
    plt.close(fig)

    # Fig. 6: heat-flow histories and the noon energy partition.
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    a = axs[0, 0]
    for k, name in [("absorbed", "Absorbed"), ("electrical", "Electrical"),
                    ("water", "Water")]:
        a.plot(hours, qw[k], lw=2, label=name)
    a = axs[0, 1]
    a.plot(hours, qw["cell_to_backplate"], lw=2, label="Cell to backplate")
    a.plot(hours, qw["backplate_to_water"], lw=2, label="Backplate to water")
    a = axs[1, 0]
    a.plot(hours, qw["front"], lw=2, label="Front loss")
    a.plot(hours, qw["rear"], lw=2, label="Rear loss")
    for a, tag, loc in zip([axs[0, 0], axs[0, 1], axs[1, 0]], "abc",
                           ["upper right", "lower center", "upper right"]):
        a.set(xlabel="Clock hour", ylabel="Power (W)", xlim=(6, 18))
        a.legend(loc=loc)
        label(a, f"({tag})")
    names = ["Electricity", "Front loss", "Rear loss", "Water heat", "Storage"]
    vals = [qw[k][noon] for k in ["electrical", "front", "rear", "water", "storage"]]
    a = axs[1, 1]
    a.barh(names, vals, color=["#2b4c7e", "#c4622d", "#a8844a", "#2a8a6e", "0.5"])
    for i, v in enumerate(vals):
        a.text(v + 8, i, f"{v:.1f} W" if v > 5 else f"{v:.2f} W", va="center")
    a.set(xlabel="Power at noon (W)", xlim=(0, max(vals) * 1.18))
    label(a, "(d)")
    fig.savefig(OUT / "fig6_heat_flows.png", dpi=DPI)
    plt.close(fig)

    # Fig. 7: node temperature gaps over the day and rates near noon.
    rates = derivatives(wet, REF_FLOW) * 60.0  # K/min.
    fig, axs = plt.subplots(2, 1, figsize=(7, 6.5), constrained_layout=True)
    axs[0].plot(hours, y[0] - y[1], lw=2, label="Cell − backplate")
    axs[0].plot(hours, y[1] - y[2], lw=2, label="Backplate − water")
    axs[0].set(xlabel="Clock hour", ylabel="Temperature difference (K)", xlim=(6, 18))
    axs[0].legend(loc="upper right")
    label(axs[0], "(a)")
    near = (hours >= 11.0) & (hours <= 13.0)
    for row, name in zip(rates, ["Cell", "Backplate", "Water"]):
        axs[1].plot(hours[near], row[near], lw=2, label=name)
    axs[1].axhline(0.0, color="0.3", lw=0.8)
    axs[1].set(xlabel="Clock hour", ylabel="Temperature rate (K/min)", xlim=(11, 13))
    axs[1].legend(loc="upper right")
    label(axs[1], "(b)")
    fig.savefig(OUT / "fig7_noon_dynamics.png", dpi=DPI)
    plt.close(fig)

    # Audit numbers quoted in the paper (Tables 6 and 8, Section 3.2).
    eig = np.sort(np.linalg.eigvals(jacobian(bm.irradiance(bm.END / 2.0), REF_FLOW)).real)
    kwh = lambda x: float(trapezoid(x, t) / 3.6e6)
    initial = np.array([bm.TA, bm.TA, bm.TIN])
    caps = np.array([bm.CC, bm.CB, bm.CW])
    partition = {}
    for name, run, q in [("dry", dry, qd), ("cooled", wet, qw)]:
        partition[name] = {
            "absorbed": kwh(q["absorbed"]), "electrical": kwh(q["electrical"]),
            "environment": kwh(q["front"] + q["rear"]), "water": kwh(q["water"]),
            "storage": float(caps @ (run[1][:, -1] - initial) / 3.6e6),
        }
    # Pointwise first law: stored rate must equal the external remainder.
    stored_rate = caps @ derivatives(wet, REF_FLOW)
    audit = {
        "noon_flux_W": {k: float(v[noon]) for k, v in qw.items()},
        "daily_partition_kWh": partition,
        "noon_eigenvalues_per_s": eig.tolist(),
        "decay_times_s": (-1.0 / eig).tolist(),
        "time_scale_ratio": float(eig.min() / eig.max()),
        "analytic_incident_kWh": 2 * bm.A * 900.0 * bm.END / np.pi / 3.6e6,
        "pointwise_balance_max_W": float(np.max(abs(stored_rate - qw["storage"]))),
        "best_tested_flow_kg_s": float(FLOWS[best]),
        "temperature_gaps_at_noon_K": {
            "cell_backplate": float(y[0, noon] - y[1, noon]),
            "backplate_water": float(y[1, noon] - y[2, noon]),
        },
    }
    (OUT / "audit_results.json").write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
