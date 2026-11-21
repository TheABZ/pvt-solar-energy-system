"""Weather-driven model of the measured PV/T collector and cooling counterfactuals.

Run after prepare_validation.py --download (needs data.csv and a C compiler):
    python weather_driven_counterfactual.py

Why this exists. The conditional benchmark (experimental_benchmark.py) uses the
measured rear-surface temperature as a boundary, so it cannot predict what the
collector would do under a different inlet temperature or with no fluid. Here
the rear boundary is replaced by ambient air (the collector has no rear
insulation), so the model needs only weather, inlet temperature and flow.
It is identified with the SAME partitions, bounds, starts and selection rule as
the benchmark, scored once on the reserved days, and then used to compare:
  dry module, measured operation, and cool-inlet operation at several flows,
all under the measured 58 days of weather.

Outputs in experimental_outputs/: weather_model.json, counterfactual_results.json,
weather_counterfactual.png
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import least_squares, lsq_linear
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experimental_benchmark as eb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "experimental_outputs"
OUT.mkdir(exist_ok=True)
DAY = 17280                     # five-second records per day
NTRAIN, NVAL = 34 * DAY, 46 * DAY
PUMP_ON_G = 50.0                # W/m2: pump runs in daylight in the scenarios
FLOWS_LPH = [10, 20, 30, 40, 50, 75, 100, 150]
PUMP_REF_W, PUMP_REF_KGS = 5.0, 0.020   # same cubic law as the controlled study


def pump_w(lph):
    """Pump power, W: 5 W x (mass flow / 0.020 kg/s)^3, as in baseline_model.py."""
    return PUMP_REF_W * (lph * eb.RHO / 3.6e6 / PUMP_REF_KGS) ** 3


def fit(fn, u, obs, ele, y0, day, active):
    """Identify both variants exactly as in the benchmark; pick on development."""
    ix = np.arange(1440, NTRAIN, 12)
    ipv, ifl = ix[day[ix]], ix[active[ix]]
    records = []
    for variant, npar in [("constant_front", 9), ("wind_front", 10)]:
        best = None
        for seed in [0, 1]:
            x0 = eb.START.copy()
            if seed:
                x0[:8] *= np.array([1.8, .6, 1.6, .6, 1.5, 1.8, .7, 1.2])
            unpack = lambda x: np.r_[np.exp(x), 0.] if npar == 9 else np.exp(x)

            def residual(x):
                z = eb.integrate(fn, u[:NTRAIN], unpack(x), ele, y0)
                return np.r_[z[ipv, 0] - obs[ipv, 0], z[ifl, 2] - obs[ifl, 1]]
            opt = least_squares(residual, np.log(x0[:npar]),
                                bounds=(np.log(eb.LOW[:npar]), np.log(eb.HIGH[:npar])),
                                max_nfev=90, ftol=2e-5, xtol=2e-5, gtol=2e-4, diff_step=1e-4)
            p = unpack(opt.x)
            cost = float(np.mean(residual(opt.x) ** 2))
            rec = dict(variant=variant, seed=seed, training_MSE=cost, nfev=int(opt.nfev),
                       parameters=dict(zip(eb.NAMES, map(float, p))),
                       active_bounds=[eb.NAMES[i] for i, v in enumerate(opt.active_mask) if v != 0],
                       jacobian_condition=float(np.linalg.cond(opt.jac)))
            records.append(rec)
            if best is None or cost < best[0]:
                best = (cost, p, rec)
        z = eb.integrate(fn, u[:NVAL], best[1], ele, y0)
        dev = np.arange(NVAL) >= NTRAIN
        pv = eb.metrics(obs[:NVAL, 0][dev & day[:NVAL]], z[dev & day[:NVAL], 0])
        fl = eb.metrics(obs[:NVAL, 1][dev & active[:NVAL]], z[dev & active[:NVAL], 2])
        best[2].update(development_PV=pv, development_fluid=fl,
                       selection_score_K=(pv["RMSE"] + fl["RMSE"]) / 2)
    chosen = min((r for r in records if "selection_score_K" in r),
                 key=lambda r: r["selection_score_K"])
    return chosen, records


def run_case(fn, u, p, ele, y0, tin=None, lph=None, dry=False):
    """Simulate one scenario; returns PV temperature, power and pump energy."""
    uu = u.copy()
    p = p.copy()
    daylight = uu[:, 0] >= PUMP_ON_G
    if dry:                                  # no heat exchanger: no fluid path
        p[6] = 1e-9
        uu[:, 3] = 0.0
        pump = np.zeros(len(uu))
    elif lph is not None:                    # constant daytime flow
        uu[:, 3] = np.where(daylight, lph * eb.RHO / 3.6e6 * eb.CP, 0.0)
        pump = np.where(daylight, pump_w(lph), 0.0)
    else:                                    # measured flow
        pump = np.zeros(len(uu))
    if tin is not None:
        uu[:, 2] = tin
    z = eb.integrate(fn, uu, p, ele, y0)
    power = np.clip(eb.power(uu[:, 0], z[:, 0], ele), 0, None)
    heat = uu[:, 3] * (z[:, 2] - uu[:, 2])
    kwh = lambda w: float(w.sum() * 5 / 3.6e6)
    return dict(z=z, power=power, gross_kWh=kwh(power), pump_kWh=kwh(pump),
                net_kWh=kwh(power) - kwh(pump), heat_kWh=kwh(np.where(daylight, heat, 0)),
                peak_PV_C=float(z[:, 0].max()))


def main():
    d, dates, u_meas, obs = eb.load()
    fn = eb.compile_solver()
    u = u_meas.copy()
    u[:, 4] = u[:, 1]          # rear boundary: ambient air, not measured rear surface
    G = u[:, 0]
    day = G >= 200
    active = day & (d.flow_rate_fluid_lph.to_numpy() > 1)
    train = np.arange(len(u)) < NTRAIN
    X = np.column_stack([G, G * (obs[:, 0] - 25), G * (G - 1000) / 1000])
    ele = lsq_linear(X[train & day], obs[train & day, 2],
                     bounds=([.01, -.01, -.3], [.8, .01, .3]), tol=1e-12).x
    y0 = np.array([obs[0, 0], (obs[0, 0] + obs[0, 1]) / 2, obs[0, 1]])

    chosen, records = fit(fn, u, obs, ele, y0, day, active)
    p = np.array([chosen["parameters"][n] for n in eb.NAMES])
    z = eb.integrate(fn, u, p, ele, y0)
    pred_power = eb.power(G, z[:, 0], ele)
    scores = {}
    for label, a, b in [("calibration", 0, NTRAIN), ("development", NTRAIN, NVAL),
                        ("held_out", NVAL, len(u))]:
        per = (np.arange(len(u)) >= a) & (np.arange(len(u)) < b)
        scores[label] = dict(
            PV_temperature_C=eb.metrics(obs[per & day, 0], z[per & day, 0]),
            flowing_outlet_C=eb.metrics(obs[per & active, 1], z[per & active, 2]),
            electrical_power_W=eb.metrics(obs[per & day, 2], pred_power[per & day]))
    (OUT / "weather_model.json").write_text(json.dumps(dict(
        rear_boundary="ambient air (measured T_air)", chosen=chosen, all_fits=records,
        electrical_coefficients=ele.tolist(), scores=scores), indent=2))

    # Counterfactuals over all 58 measured days with the frozen weather-driven model.
    ta = u[:, 1]
    cases = {"dry_module": run_case(fn, u, p, ele, y0, dry=True),
             "measured_operation": run_case(fn, u, p, ele, y0)}
    for lph in FLOWS_LPH:
        cases[f"cool_inlet_{lph}Lph"] = run_case(fn, u, p, ele, y0, tin=ta, lph=lph)
    cases["hot_measured_inlet_50Lph"] = run_case(fn, u, p, ele, y0, lph=50)
    dry = cases["dry_module"]
    ndays = len(u) / DAY
    summary = {}
    for k, c in cases.items():
        summary[k] = dict(gross_kWh=c["gross_kWh"], pump_kWh=c["pump_kWh"], net_kWh=c["net_kWh"],
                          heat_kWh=c["heat_kWh"], peak_PV_C=c["peak_PV_C"],
                          net_gain_vs_dry_pct=100 * (c["net_kWh"] / dry["gross_kWh"] - 1),
                          gross_gain_vs_dry_pct=100 * (c["gross_kWh"] / dry["gross_kWh"] - 1),
                          mean_daily_net_kWh=c["net_kWh"] / ndays)
    illum = G >= 600
    # Sensitivity: typical crystalline-silicon coefficient (-0.4 %/K) instead of fitted.
    ele_typ = ele.copy()
    ele_typ[1] = -0.004 * ele[0]
    dry_t = run_case(fn, u, p, ele_typ, y0, dry=True)
    sens = {}
    for lph in [20, 50, 100]:
        c = run_case(fn, u, p, ele_typ, y0, tin=ta, lph=lph)
        sens[f"cool_inlet_{lph}Lph"] = dict(
            gross_gain_vs_dry_pct=100 * (c["gross_kWh"] / dry_t["gross_kWh"] - 1),
            net_gain_vs_dry_pct=100 * (c["net_kWh"] / dry_t["gross_kWh"] - 1))
    m = run_case(fn, u, p, ele_typ, y0)
    sens["measured_operation"] = dict(gross_gain_vs_dry_pct=100 * (m["gross_kWh"] / dry_t["gross_kWh"] - 1))
    summary["sensitivity_beta_0p004"] = sens
    summary["_notes"] = dict(
        days=ndays, pump_law=f"{PUMP_REF_W} W x (mass flow/{PUMP_REF_KGS} kg/s)^3 in daylight (G >= {PUMP_ON_G} W/m2)",
        fitted_relative_temperature_coefficient_per_K=float(ele[1] / ele[0]),
        mean_PV_C_at_G_ge_600={k: float(c["z"][illum, 0].mean()) for k, c in cases.items()},
        measured_mean_PV_C_at_G_ge_600=float(obs[illum, 0].mean()),
        measured_mean_inlet_C_when_flowing_G_ge_600=float(u[illum & (u[:, 3] > 0), 2].mean()))
    (OUT / "counterfactual_results.json").write_text(json.dumps(summary, indent=2))

    # Figure: net and gross gain versus flow, cool inlet, with dry and hot-inlet marks.
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    g = [summary[f"cool_inlet_{f}Lph"]["gross_gain_vs_dry_pct"] for f in FLOWS_LPH]
    n = [summary[f"cool_inlet_{f}Lph"]["net_gain_vs_dry_pct"] for f in FLOWS_LPH]
    ax.plot(FLOWS_LPH, g, "o-", color="#1f3b73", lw=2, label="Gross, inlet at ambient")
    ax.plot(FLOWS_LPH, n, "o-", color="#c46a1b", lw=2, label="Net of pump, inlet at ambient")
    ax.axhline(0, color="0.4", ls="--", lw=1.2, label="Dry module")
    ax.plot([50], [summary["hot_measured_inlet_50Lph"]["net_gain_vs_dry_pct"]], "s",
            color="#a33", ms=7, label="Net, measured hot inlet, 50 L/h")
    ax.set_ylim(-6, 2.5)
    ax.set(xlabel="Daytime fluid flow (L/h)", ylabel="Electricity change vs dry (%)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.savefig(OUT / "weather_counterfactual.png", dpi=300)
    plt.close(fig)
    print(json.dumps(dict(chosen_variant=chosen["variant"], active_bounds=chosen["active_bounds"],
                          parameters=chosen["parameters"], scores=scores,
                          summary=summary), indent=1))


if __name__ == "__main__":
    main()
