#!/usr/bin/env python3
"""Summarise the best fit of a set: parameters, logpost terms, topology, energy plausibility, figure.

usage: summarize_fit.py <setname> <fit_json> [<fit_json> ...]   (env: espei-pr268)
Writes results/<setname>_best.json, figures/fit_<setname>_best.png and a ranked list of all endpoints.
Energy plausibility (reported, not a gate): excess enthalpy and entropy at x_Ni = 0.25/0.5/0.75 of each
phase vs the data (DFT SQS mixing enthalpies, Dench/Watson HM_FORM-derived, liquid HM_MIX), |b| terms,
and spurious stability at 300-800 K (outside the topology window).
"""
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni
import pdfast


def hmix(th, x):
    """excess enthalpy (J/mol) and entropy (J/mol/K) of LIQ, BCC, FCC at x_Ni (a+bT model)."""
    xc, xn = 1 - x, x
    d = xc - xn
    H = {"LIQUID": xc * xn * th[0], "BCC_A2": xc * xn * (th[2] + th[4] * d), "FCC_A1": xc * xn * (th[6] + th[8] * d)}
    S = {"LIQUID": -xc * xn * th[1], "BCC_A2": -xc * xn * (th[3] + th[5] * d), "FCC_A1": -xc * xn * (th[7] + th[9] * d)}
    return H, S


def main():
    setname = sys.argv[1]
    res = []
    for f in sys.argv[2:]:
        if Path(f).exists():
            res += json.loads(Path(f).read_text())
    res = [r for r in res if np.isfinite(r["logpost"])]
    res.sort(key=lambda r: -r["logpost"])
    T = np.arange(800.0, 2201.0, 5.0)
    ref = pdfast.ref_grid(T, crni.TDB)
    Tlow = np.arange(300.0, 800.0, 10.0)
    reflow = pdfast.ref_grid(Tlow, crni.TDB)
    ranked = []
    for r in res:
        th = np.asarray(r["theta"])
        top = pdfast.topology(th, T, ref)
        ranked.append({"start": r["start"], "logpost": r["logpost"], "grad_norm_q": r["grad_norm_q"],
                       "topology_ok": top["ok"], "T_liquid_min": top["T_liquid_min"], "n_bad_T": top["n_bad"],
                       "gaps": len(top["gaps"]), "messages": r.get("messages"),
                       "nonfinite_events": len(r.get("nonfinite_events", [])), "theta": r["theta"]})
    # ranking: topology first (with plausible energies), then logpost
    ok = [z for z in ranked if z["topology_ok"]]
    best = (ok or ranked)[0]
    th = np.asarray(best["theta"])
    top = pdfast.topology(th, T, ref)
    low = [pdfast.seq_string(s) for s in pdfast.diagram(th, Tlow, reflow)]
    low_seqs = sorted({" -> ".join(s) for s in low})
    b = pdfast.boundaries(th, np.array([800.0, 1000.0, 1200.0, 1473.0]),
                          pdfast.ref_grid(np.array([800.0, 1000.0, 1200.0, 1473.0]), crni.TDB))
    en = {}
    for x in (0.25, 0.5, 0.75):
        H, S = hmix(th, x)
        en[f"x_Ni={x}"] = {"H_ex_J": {k: round(v, 1) for k, v in H.items()}, "S_ex_J_K": {k: round(v, 3) for k, v in S.items()}}
    fig = crni.RUN / "figures" / f"fit_{setname}_best.png"
    tmp = crni.RUN / "fit" / f"{setname}_best_theta.json"
    tmp.write_text(json.dumps({"theta": best["theta"]}))
    subprocess.run([sys.executable, str(crni.RUN / "scripts" / "plotpd.py"), str(tmp), str(fig),
                    f"{setname}: best fit (start {best['start']}, logpost {best['logpost']:.1f})"], check=True,
                   capture_output=True)
    out = {"set": setname, "best_start": best["start"], "logpost": best["logpost"], "grad_norm_q": best["grad_norm_q"],
           "theta": dict(zip(crni.LABELS, best["theta"])),
           "tdb_symbols": dict(zip(crni.NAMES, best["theta"])),
           "topology_800_2200K": {"ok": top["ok"], "T_liquid_min_K": top["T_liquid_min"],
                                  "T_bcc_fcc_max_K": top["T_bcc_fcc_max"], "gaps": top["gaps"][:5], "bad_T": top["bad"][:5]},
           "sequences_300_800K": low_seqs,
           "boundaries_xNi": {f"{t:.0f}K": {"bcc": float(b['bcc_bf'][i]), "fcc": float(b['fcc_bf'][i])}
                              for i, t in enumerate([800, 1000, 1200, 1473])},
           "zhang2014_1473K": {"bcc": 0.282, "fcc": 0.508},
           "excess_energies": en,
           "reference_values": {"DFT_HM_MIX_kJ (Liu 2016, x_Cr .75/.5/.25)": {"BCC": [10.02, 9.15, 4.98], "FCC": [-12.69, -9.03, -5.79]},
                                "liquid_HM_MIX_x0.5_kJ": "about -3 to -4.9 (Saltykov, Thiedemann, Sudavtsova); Batalin positive"},
           "paper_opt_for_comparison": dict(zip(crni.LABELS, crni.PAPER_OPT.tolist())),
           "figure": str(fig.relative_to(crni.RUN)),
           "any_endpoint_topology_ok": bool(ok),
           "ranked_endpoints": [{k: v for k, v in z.items() if k != "theta"} | {"theta": z["theta"]} for z in ranked]}
    (crni.RUN / "results" / f"{setname}_best.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("set", "best_start", "logpost", "grad_norm_q", "topology_800_2200K",
                                          "sequences_300_800K", "boundaries_xNi")}, indent=1))
    print(json.dumps(out["excess_energies"]))
    for z in ranked:
        print(f"{z['start']:22s} {z['logpost']:10.2f} g={z['grad_norm_q']:8.3g} topo={z['topology_ok']} "
              f"Tliq={z['T_liquid_min']:.0f} gaps={z['gaps']} nonfinite={z['nonfinite_events']}")


if __name__ == "__main__":
    main()
