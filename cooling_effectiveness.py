"""Closed-form cooling effectiveness and critical inlet temperature for a PV/T network.

Run:  python cooling_effectiveness.py     (needs baseline_model.py; the measured-
collector part also reads experimental_outputs/weather_model.json if present)
Writes figures/fig11_cooling_effectiveness_map.png and
experimental_outputs/cooling_effectiveness.json (Section 2.14, Table 14, Fig. 11).

Quasi-steady reduction of the three-node network (cell c, backplate b, fluid w):
  front loss H_f (cell -> ambient), H_cb (cell -> backplate), H_ba (backplate ->
  ambient), H_bw (backplate -> fluid), M = m_dot*c_p (fluid -> inlet).
Eliminating the fluid gives H_eff = H_bw*M/(H_bw+M) from backplate to inlet.
Eliminating the backplate gives a single path from the cell with conductance
  K = H_cb*(H_ba+H_eff)/(H_cb+H_ba+H_eff)
to a sink at T* = Ta + phi*(Tin-Ta), phi = H_eff/(H_ba+H_eff). Then
  Tc - Ta = [q + K*phi*(Tin-Ta)] / (H_f + K),  q = absorbed heat (W).
With K_d = H_cb*H_ba/(H_cb+H_ba) for the dry module:
  cooling effectiveness  E = (K_w - K_d)/(H_f + K_w)     (inlet at ambient)
    = fraction of the dry excess temperature (Tc,dry - Ta) removed by the fluid;
  critical inlet temperature, above which circulation no longer cools the cell:
  Tin,crit = Ta + (Tc,dry - Ta) * (K_w - K_d)/(phi*K_w)
           = Ta + (Tc,dry - Ta) * H_cb/(H_cb + H_ba) = T_b,dry,
  i.e. the backplate temperature of the dry module, independent of flow.
For any inlet temperature the cell cooling is
  Tc,dry - Tc,wet = F_c * (T_b,dry - Tin),  F_c = phi*K_w/(H_f + K_w),
where F_c is a cell-to-fluid coupling factor between 0 and 1.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import baseline_model as bm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)


def network(H_f, H_cb, H_ba, H_bw, M):
    """Return K_d, K_w, phi, effectiveness E for one operating point."""
    H_eff = H_bw * M / (H_bw + M) if M > 0 else 0.0
    K_d = H_cb * H_ba / (H_cb + H_ba)
    K_w = H_cb * (H_ba + H_eff) / (H_cb + H_ba + H_eff)
    phi = H_eff / (H_ba + H_eff) if H_eff > 0 else 0.0
    return K_d, K_w, phi, (K_w - K_d) / (H_f + K_w)


def coupling(H_f, K_w, phi):
    """Cell-to-fluid coupling factor F_c = phi*K_w/(H_f + K_w)."""
    return phi * K_w / (H_f + K_w)


def tin_crit(Tc_dry, Ta, K_d, K_w, phi):
    return Ta + (Tc_dry - Ta) * (K_w - K_d) / (phi * K_w)


def controlled_checks():
    """Compare the closed form with the transient model at noon."""
    G = 900.0
    rows = []
    # open rack: rear loss equal to the front loss (19.2 W/K), as in Section 2.7 of the paper
    for label, hba in [("rear-insulated", bm.HBA), ("open-rack", bm.HCA)]:
        dry = bm.simulate(0.0, hba=hba)
        j = int(np.argmin(abs(dry[0] - bm.END / 2)))
        Tc_dry_sim = dry[1][0, j]
        for m in [0.005, 0.0125, 0.020, 0.040]:
            wet = bm.simulate(m, hba=hba)
            Tc_sim = wet[1][0, j]
            M = m * bm.CP
            K_d, K_w, phi, E = network(bm.HCA, bm.HCB, hba, bm.HBW, M)
            # closed form uses the simulated dry excess and predicts the cooled cell
            Tc_pred = bm.TA + (1 - E) * (Tc_dry_sim - bm.TA)
            E_sim = (Tc_dry_sim - Tc_sim) / (Tc_dry_sim - bm.TA)
            rows.append(dict(case=label, flow_kg_s=m, E_closed=E, E_transient=float(E_sim),
                             Tc_pred_C=float(Tc_pred), Tc_transient_C=float(Tc_sim),
                             F_c=coupling(bm.HCA, K_w, phi),
                             Tin_crit_C=float(tin_crit(Tc_dry_sim, bm.TA, K_d, K_w, phi))))
        # transient check of the threshold: inlet held at Tin,crit, 0.020 kg/s
        K_d, K_w, phi, _ = network(bm.HCA, bm.HCB, hba, bm.HBW, 0.020 * bm.CP)
        tc = tin_crit(Tc_dry_sim, bm.TA, K_d, K_w, phi)
        tin0 = bm.TIN
        bm.TIN = tc
        try:
            hot = bm.simulate(0.020, hba=hba)
        finally:
            bm.TIN = tin0
        rows.append(dict(case=label + " at Tin,crit", flow_kg_s=0.020, Tin_C=float(tc),
                         Tc_dry_noon_C=float(Tc_dry_sim), Tc_noon_C=float(hot[1][0, j]),
                         Tb_dry_noon_C=float(dry[1][1, j])))
    return rows


def measured_collector():
    """Effectiveness and critical inlet for the identified weather-driven model."""
    f = ROOT / "experimental_outputs" / "weather_model.json"
    if not f.exists():
        return None
    p = json.loads(f.read_text())["chosen"]["parameters"]
    rho, cp = 1040.0, 3650.0
    v_mean = 2.31   # mean wind speed (m/s) in illuminated flowing operation, G >= 600 W/m2
    H_f = p["H_front"] + p["H_wind"] * v_mean + p["H_sky"]
    out = []
    for lph in [10, 20, 50, 100]:
        M = lph * rho / 3.6e6 * cp
        K_d, K_w, phi, E = network(H_f, p["H_PV_absorber"], p["H_absorber_rear"],
                                   p["H_absorber_fluid"], M)
        out.append(dict(flow_Lph=lph, E=E, phi=phi, K_d=K_d, K_w=K_w, F_c=coupling(H_f, K_w, phi),
                        crit_fraction=(K_w - K_d) / (phi * K_w)))
    return dict(H_f_W_K=H_f, H_cb=p["H_PV_absorber"], H_ba=p["H_absorber_rear"],
                H_bw=p["H_absorber_fluid"], rows=out)


def design_map(path, markers):
    """Cooling effectiveness E over the two conductance ratios that set it."""
    x = np.logspace(np.log10(0.3), np.log10(20), 160)    # H_cb / H_f
    y = np.logspace(np.log10(0.1), np.log10(10), 160)    # H_eff / H_f
    X, Y = np.meshgrid(x, y)
    hba = 0.1                                            # H_ba / H_f (fixed slice)
    Kd = X * hba / (X + hba)
    Kw = X * (hba + Y) / (X + hba + Y)
    E = (Kw - Kd) / (1 + Kw)
    fig, ax = plt.subplots(figsize=(7, 4.6), constrained_layout=True)
    cs = ax.contourf(X, Y, E, levels=np.linspace(0, 0.9, 19), cmap="viridis")
    c = ax.contour(X, Y, E, levels=[0.1, 0.2, 0.4, 0.6, 0.8], colors="w", linewidths=0.8)
    ax.clabel(c, fmt="%.1f", fontsize=8)
    for lab, mx, my, mk in markers:
        ax.plot(mx, my, mk, color="#ff6f3c", ms=8, mec="k")
        right = mx > 3
        ax.annotate(lab, (mx, my), xytext=(-8, 8) if right else (8, 8), textcoords="offset points",
                    color="w", fontsize=8, ha="right" if right else "left")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set(xlabel="Cell-to-absorber conductance / front loss, $H_{cb}/H_f$",
           ylabel="Absorber-to-inlet conductance / front loss, $H_{eff}/H_f$")
    fig.colorbar(cs, ax=ax, label="Cooling effectiveness $E$ (inlet at ambient)")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    res = dict(controlled=controlled_checks(), measured_collector=measured_collector())
    hf, M = bm.HCA, 0.020 * bm.CP
    heff = bm.HBW * M / (bm.HBW + M)
    markers = [("Controlled case, 0.020 kg/s", bm.HCB / hf, heff / hf, "o")]
    mc = res["measured_collector"]
    if mc:
        for r in mc["rows"]:
            if r["flow_Lph"] in (20, 50):
                he = mc["H_bw"] * (r["flow_Lph"] * 1040 / 3.6e6 * 3650) / (mc["H_bw"] + r["flow_Lph"] * 1040 / 3.6e6 * 3650)
                markers.append((f"Measured collector, {r['flow_Lph']} L/h", mc["H_cb"] / mc["H_f_W_K"], he / mc["H_f_W_K"], "s"))
    res["map_markers"] = [dict(label=a, x=b, y=c) for a, b, c, _ in markers]
    design_map(OUT / "fig11_cooling_effectiveness_map.png", markers)
    res_dir = ROOT / "experimental_outputs"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "cooling_effectiveness.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
