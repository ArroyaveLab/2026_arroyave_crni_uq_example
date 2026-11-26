"""Write the summary of the second fitted point of set E (the one with a stable BCC+BCC field):
its parameter vector, phase-field sequence and figure. Run from the repository root after fit.py."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import numpy as np, crni, pdfast, subprocess
best = json.load(open("results/setE_orchid_sigmas_best.json"))
gap = [z for z in best["ranked_endpoints"] if not z["topology_ok"]]
gap.sort(key=lambda z: -z["logpost"]); g = gap[0]; th = np.asarray(g["theta"])
T = np.arange(800.0, 2201.0, 5.0); top = pdfast.topology(th, T, pdfast.ref_grid(T, crni.TDB))
split = json.load(open("results/bcc_split_energy_check.json"))["gapmode"]
conv = json.load(open("results/bcc_convexity_check.json"))
fig = "figures/fit_setE_orchid_sigmas_gapmode.png"
json.dump({"theta": g["theta"]}, open("fit/setE_gapmode_best_theta.json", "w"))
subprocess.run([sys.executable, "scripts/plotpd.py", "fit/setE_gapmode_best_theta.json", fig,
                f"setE gap mode (start {g['start']}, logpost {g['logpost']:.2f})"], check=True, capture_output=True)
out = {"set": "setE_orchid_sigmas", "mode": "BCC+BCC gap (best wrong-topology endpoint)", "start": g["start"],
       "logpost": g["logpost"], "grad_norm_q": g["grad_norm_q"], "theta": dict(zip(crni.LABELS, g["theta"])),
       "tdb_symbols": dict(zip(crni.NAMES, g["theta"])),
       "topology_800_2200K": {"ok": top["ok"], "T_liquid_min_K": top["T_liquid_min"], "gaps_first": top["gaps"][:6], "n_bad_T": top["n_bad"]},
       "endpoints_in_this_mode": [(z["start"], z["logpost"]) for z in gap],
       "logpost_gap_vs_correct": g["logpost"] - best["logpost"],
       "bcc_split_energy_at_observed_states (theta of near_opt_0, same mode)": split,
       "bcc_spinodal_top_K_x": conv["Tc_fit_gapmode"], "figure": fig,
       "note": "No source we hold reports a BCC gap. ESPEI single-phase BCC observations do not penalise the split (driving force 0 J at the 27 states inside the gap region); split energies at observed states <= 121 J/mol-atom."}
json.dump(out, open("results/setE_orchid_sigmas_gapmode.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("start", "logpost", "logpost_gap_vs_correct", "endpoints_in_this_mode")}, indent=1))
