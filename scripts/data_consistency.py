#!/usr/bin/env python3
"""Data consistency check (orchid-sage docs/fit_protocol.md 0.2a) for every Cr-Ni dataset of a set. Flags only.

usage: data_consistency.py <setname> <out_json> name=theta.json|paper_opt ...   (env: espei-pr268)
Items: (1) sigma and basis per observation; (2) within-series jumps, monotonicity, physical bounds;
(3) partial vs integral enthalpy (liquid, model-free); (4) basis and units; (5) duplicates;
(6) cross-source disagreements at the same state; (7) per-dataset and per-point pull at the given parameters.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import crni

R = 8.314462618
setname, outp = sys.argv[1], Path(sys.argv[2])
pts = {}
for a in sys.argv[3:]:
    n, v = a.split("=")
    pts[n] = crni.PAPER_OPT if v == "paper_opt" else np.asarray(json.loads(Path(v).read_text())["theta"])
DS = sorted(crni.datasets_dir(setname).glob("*.json"))
data = {f.name: json.loads(f.read_text()) for f in DS}
out = {"set": setname, "items": {}}


def xni_list(d):
    if d["output"].startswith("HM"):
        return [occ[0][1] for occ in d["solver"]["sublattice_occupancies"]]
    c = d["conditions"]
    return list(c["X_NI"]) if "X_NI" in c else [1 - v for v in c["X_CR"]]


def sigma_of(name, d):
    w = d.get("weight", 1.0)
    if d["output"] == "ZPF":
        return 1000.0 / w, "ZPF driving-force scale 1000 J/mol / weight (ORCHID rule: ZPF weight left at 1; not a value sigma)"
    s = 500.0 / w
    c = d.get("comment", "")
    if name.startswith("dft_"):
        basis = "ORCHID DFT sigma_effective 7.2 kJ/mol-atom"
    elif "set E: ORCHID-rule" in c:
        basis = "ORCHID rule, method class 'unknown' floor (set E)"
    elif "ORCHID" in c or name.startswith("hao_") or "same measurement as ORCHID" in c:
        basis = "ORCHID record sigma_effective (Hao export)"
    else:
        basis = "SOFTWARE DEFAULT (ESPEI 500 J/mol) - flag"
    return s, basis


# (1)
it1 = {}
for n, d in data.items():
    s, b = sigma_of(n, d)
    nv = len(d["values"]) if d["output"] == "ZPF" else len(np.ravel(d["values"]))
    it1[n] = {"output": d["output"], "n": nv, "sigma_J": round(s, 2), "basis": b, "flag": "DEFAULT" in b}
out["items"]["1_sigma"] = it1

# (2) within-series
it2 = {}
for n, d in data.items():
    if d["output"] == "ZPF" or n.startswith("dft_"):
        continue
    x = np.array(xni_list(d), float)
    y = np.ravel(d["values"]).astype(float)
    T = float(d["conditions"]["T"])
    s, _ = sigma_of(n, d)
    flags = []
    if d["output"].startswith("ACR"):
        if np.any((y <= 0) | (y > 1)):
            flags.append({"check": "bounds (0,1]", "indices": np.where((y <= 0) | (y > 1))[0].tolist()})
        # jumps tested on RT ln gamma_Cr (ln a falls steeply as x_Cr -> 0, which is not a jump)
        mu = R * T * np.log(np.clip(y, 1e-12, None) / np.clip(1 - x, 1e-12, None))
        o = np.argsort(x)
        xs, ms = x[o], mu[o]
        for k in range(len(xs)):
            nb = [ms[j] for j in (k - 1, k + 1) if 0 <= j < len(xs)]
            if len(nb) == 2 and min(abs(ms[k] - nb[0]), abs(ms[k] - nb[1])) > 3 * s:
                flags.append({"check": "jump vs both neighbours > 3 sigma (in RT ln a)", "x_Ni": float(xs[k]),
                              "a_Cr": float(y[o][k]), "neighbours_a": [float(y[o][j]) for j in (k - 1, k + 1)],
                              "deviation_J": float(min(abs(ms[k] - nb[0]), abs(ms[k] - nb[1])))})
            elif len(nb) == 1 and abs(ms[k] - nb[0]) > 3 * s:
                flags.append({"check": "end-point jump > 3 sigma", "x_Ni": float(xs[k]), "a_Cr": float(y[o][k])})
        lna = R * T * np.log(np.clip(y[o], 1e-12, None))
        dec = [(float(xs[k]), float(y[o][k]), float(y[o][k + 1])) for k in range(len(xs) - 1)
               if lna[k + 1] - lna[k] > 2 * s]  # a_Cr must not rise with x_Ni
        if dec:
            flags.append({"check": "a_Cr rises with x_Ni by > 2 sigma (non-monotonic)", "where": dec})
        lo = int(np.argmin(x))
        flags.append({"check": "a_Cr -> 1 as x_Cr -> 1 (info)", "smallest_x_Ni": float(x[lo]), "a_Cr": float(y[lo])})
    else:
        o = np.argsort(x)
        xs, ys = x[o], y[o]
        for k in range(1, len(xs) - 1):
            mid = ys[k - 1] + (ys[k + 1] - ys[k - 1]) * (xs[k] - xs[k - 1]) / max(xs[k + 1] - xs[k - 1], 1e-9)
            if abs(ys[k] - mid) > 3 * s:
                flags.append({"check": "jump vs neighbours > 3 sigma", "x_Ni": float(xs[k]), "value": float(ys[k])})
        if d["output"] == "HM_MIX":
            pos = (y > 0).sum()
            if 0 < pos < len(y):
                flags.append({"check": "sign change within a mixing-enthalpy series", "n_positive": int(pos),
                              "x_Ni_positive": [float(v) for v in x[y > 0]]})
    it2[n] = flags
out["items"]["2_within_series"] = it2


# (3) partial vs integral (liquid): partial H_Cr from d(R ln gamma_Cr)/d(1/T) (Katayama 1733/1783/1833 K, same x)
# against the partial H_Cr implied by the integral liquid HM_MIX series (quadratic RK fit per series)
def rk_fit(x, h):
    xc = 1 - x
    A = np.stack([x * xc, x * xc * (xc - x)], axis=1)
    c, *_ = np.linalg.lstsq(A, h, rcond=None)
    return c


def partial_cr(c, x):
    xc = 1 - x
    H = x * xc * (c[0] + c[1] * (xc - x))
    dH_dxc = (xc * (1 - 2 * xc) * 0 + (x - xc)) * (c[0] + c[1] * (xc - x)) + x * xc * c[1] * 2  # d/dxc with x = 1 - xc
    return H + x * dH_dxc


kat = {n: d for n, d in data.items() if "ACR_CR-LIQUID-kat" in n}
it3 = {"method": "partial H_Cr = -R d ln(gamma_Cr)/d(1/T) [sign: d(R ln a)/d(1/T) = H_bar], 3 temperatures, "
                 "vs H_bar_Cr = H + x_Ni dH/dx_Cr from an RK2 fit of each liquid HM_MIX series", "rows": []}
if len(kat) >= 2:
    series = []
    for n, d in kat.items():
        series.append((float(d["conditions"]["T"]), np.array(xni_list(d)), np.ravel(d["values"]).astype(float)))
    xs_common = [0.5, 0.6, 0.7, 0.8, 0.9]
    hm = {n: rk_fit(np.array(xni_list(d)), np.ravel(d["values"]).astype(float))
          for n, d in data.items() if d["output"] == "HM_MIX" and d["phases"] == ["LIQUID"]}
    for xq in xs_common:
        invT, Rlna = [], []
        for T, x, a in series:
            k = np.argmin(abs(x - xq))
            if abs(x[k] - xq) < 0.01:
                invT.append(1 / T); Rlna.append(R * np.log(a[k] / (1 - x[k])))
        if len(invT) >= 2:
            slope = np.polyfit(invT, Rlna, 1)[0]
            it_ = np.asarray(invT)
            s_slope = (2900.0 / (1.0 / np.mean(it_))) / np.sqrt(np.sum((it_ - it_.mean()) ** 2))
            row = {"x_Ni": xq, "H_bar_Cr_from_activity_T_dependence_J": float(slope), "sigma_J": float(s_slope),
                   "n_T": len(invT),
                   "from_integral_HM_MIX_J": {n.split("LIQUID-")[1][:20]: float(partial_cr(c, xq)) for n, c in hm.items()}}
            it3["rows"].append(row)
    it3["note"] = ("Three temperatures 100 K apart: the activity-derived partial enthalpy has an uncertainty of order "
                   "sigma_mu * sqrt(2) / (R-scaled 1/T span) ~ tens of kJ/mol, so only gross disagreement is informative.")
out["items"]["3_partial_vs_integral"] = it3

# (4) basis and units
it4 = {}
for n, d in data.items():
    c = d.get("comment", "")
    it4[n] = {"output": d["output"], "reference_state": d.get("reference_state", "phase end-members (HM_MIX) / SER (HM_FORM)"
              if d["output"].startswith("HM") else None),
              "basis_note": ("per mole of atoms ASSUMED (Hao basis being checked)" if n.startswith("hao_CR-NI-HM_MIX") else
                             "kJ/mol-atom x1000; 0 K DFT entered at 298.15 K" if n.startswith("dft_") else
                             "cal->J conversion by the dataset author" if "cal/mol" in c else
                             "wt%->at% conversion by the dataset author" if "weight percent" in c else ""),
              "T_K": d["conditions"].get("T") if not isinstance(d["conditions"].get("T"), list) else
              [min(d["conditions"]["T"]), max(d["conditions"]["T"])]}
out["items"]["4_basis_units"] = it4

# (5) duplicates
it5 = [
    {"what": "Saltykov 2002 and Thiedemann 1998 liquid HM_MIX in ESPEI-datasets AND in the Hao 2021 figure (ORCHID export)",
     "treatment": "used once (ESPEI-datasets digitisation), ORCHID sigma"},
    {"what": "Svechnikov 1962 ZPF digitised from Tang & Hallstedt 2016 Fig. 1 (secondary; the original was not obtained)",
     "treatment": "used once"},
    {"what": "Hao 2021 Fig. 16 ZPF marks (ORCHID export c16a/c16b) re-plot Collins 1988, Dench 1963, Karmazin 1982 and others",
     "treatment": "not used (the ESPEI-datasets digitisations of the originals are used)"},
    {"what": "set D/E: ZPF regions with one known end enter as (i) single-phase and (ii) null-vertex two-phase observations",
     "treatment": "two observations of one parent point (rule 0.4a); no further double counting claimed"},
]
jt = data.get("zpf_CR-NI-ZPF-BCC_A2-FCC_A1-jette1934xray.json")
if jt:
    seen, dup = {}, []
    for i, (t, r) in enumerate(zip(jt["conditions"]["T"], jt["values"])):
        k = (t, tuple((v[0], v[2][0]) for v in r))
        if k in seen:
            dup.append([seen[k], i])
        seen.setdefault(k, i)
    it5.append({"what": "Jette 1934: identical (T, phase, composition) rows within the dataset ('both compositions included')",
                "pairs": dup})
out["items"]["5_duplicates"] = it5

# (6) cross-source at the same state
it6 = {"liquid_HM_MIX": [], "liquid_ACR_1873K": [], "solid_HM_FORM": [], "ZPF_FCC_boundary": []}
hmL = {n.split("LIQUID-")[1][:25]: (np.array(xni_list(d)), np.ravel(d["values"]).astype(float), sigma_of(n, d)[0])
       for n, d in data.items() if d["output"] == "HM_MIX" and d["phases"] == ["LIQUID"]}
for xq in (0.55, 0.65, 0.75, 0.85, 0.95):
    vals = {}
    for k, (x, h, s) in hmL.items():
        o = np.argsort(x)
        if x.min() - 0.02 <= xq <= x.max() + 0.02:
            vals[k] = float(np.interp(xq, x[o], h[o]))
    if len(vals) > 1:
        spread = max(vals.values()) - min(vals.values())
        it6["liquid_HM_MIX"].append({"x_Ni": xq, "values_J": vals, "spread_J": spread,
                                     "spread_over_sigma_pair": spread / (1000 * np.sqrt(2)), "flag": spread > 2 * 1000 * np.sqrt(2)})
acr = {n.split("LIQUID-")[1][:12]: (np.array(xni_list(d)), np.ravel(d["values"]).astype(float))
       for n, d in data.items() if "ACR_CR-LIQUID" in n and float(d["conditions"]["T"]) == 1873.0}
for xq in (0.55, 0.6, 0.7, 0.8, 0.85):
    vals = {k: float(np.interp(xq, x[np.argsort(x)], a[np.argsort(x)])) for k, (x, a) in acr.items()
            if x.min() - 0.02 <= xq <= x.max() + 0.02}
    if len(vals) > 1:
        mu = {k: R * 1873 * np.log(v) for k, v in vals.items()}
        sp = max(mu.values()) - min(mu.values())
        it6["liquid_ACR_1873K"].append({"x_Ni": xq, "a_Cr": vals, "spread_mu_J": sp,
                                        "spread_over_sigma_pair": sp / (2900 * np.sqrt(2)), "flag": sp > 2 * 2900 * np.sqrt(2)})
hf = {}
for n, d in data.items():
    if d["output"] == "HM_FORM":
        for x, v in zip(xni_list(d), np.ravel(d["values"])):
            hf.setdefault((d["phases"][0], round(x, 1)), []).append((n.split("HM_FORM-")[1].split("-")[1][:10], float(v),
                                                                     float(d["conditions"]["T"])))
for (ph, x), rows in sorted(hf.items()):
    srcs = {r[0] for r in rows}
    if len(srcs) > 1:
        vals = [r[1] for r in rows]
        sp = max(vals) - min(vals)
        it6["solid_HM_FORM"].append({"phase": ph, "x_Ni~": x, "rows": rows, "spread_J": sp,
                                     "flag": sp > 2 * 3500 * np.sqrt(2)})
fcc = []
for n, d in data.items():
    if d["output"] != "ZPF" or n.startswith("zpf_single"):
        continue
    Tl = d["conditions"]["T"] if isinstance(d["conditions"]["T"], list) else [d["conditions"]["T"]] * len(d["values"])
    for t, r in zip(Tl, d["values"]):
        ph = {v[0] for v in r}
        if ph == {"BCC_A2", "FCC_A1"}:
            for v in r:
                if v[0] == "FCC_A1" and v[2][0] is not None:
                    fcc.append((d["reference"], float(t), v[2][0] if v[1] == ["NI"] else 1 - v[2][0]))
for src in sorted({f[0] for f in fcc}):
    mine = [f for f in fcc if f[0] == src]
    others = [f for f in fcc if f[0] != src]
    dev = []
    for _, t, x in mine:
        near = [o[2] for o in others if abs(o[1] - t) <= 30]
        if near:
            dev.append(x - float(np.median(near)))
    if dev:
        it6["ZPF_FCC_boundary"].append({"source": src, "n": len(dev), "median_dx_vs_others_within_30K": float(np.median(dev)),
                                        "flag": abs(float(np.median(dev))) > 2 * 0.011 * np.sqrt(2)})
out["items"]["6_cross_source"] = it6

# (7) per-dataset and per-point pull
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from pycalphad import Database
it7 = {"per_dataset": {}, "dominating_points": []}
for f in DS:
    ds = load_datasets([str(f)])
    ctx = setup_context(Database(str(crni.TDB)), ds, symbols_to_fit=crni.NAMES, phase_models=crni.phase_models())
    row = {}
    for pn, th in pts.items():
        tot, contrib = 0.0, None
        for r in ctx["residual_objs"]:
            v = float(r.get_likelihood(th))
            tot += v
            try:
                res, w = r.get_residuals(th)
                res, w = np.asarray(res, float), np.asarray(w, float)
                if res.size:
                    scale = 1000.0 / w if type(r).__name__ == "ZPFResidual" else w  # ZPF returns weights, others std
                    contrib = 0.5 * (res / scale) ** 2
            except Exception:  # noqa
                pass
        row[pn] = {"loglike": tot}
        if contrib is not None and contrib.size:
            share = contrib / max(contrib.sum(), 1e-30)
            row[pn].update({"n": int(contrib.size), "chi2_half": float(contrib.sum()),
                            "per_point_mean": float(contrib.mean()), "max_point_share": float(share.max()),
                            "max_point_index": int(np.argmax(share))})
            if share.max() > 0.5 and contrib.sum() > 5:
                it7["dominating_points"].append({"dataset": f.name, "at": pn, "index": int(np.argmax(share)),
                                                 "share": float(share.max()), "chi2_half": float(contrib.max())})
    it7["per_dataset"][f.name] = row
    print(f.name, {k: round(v["loglike"], 1) for k, v in row.items()}, flush=True)
out["items"]["7_pull"] = it7
outp.write_text(json.dumps(out, indent=1, default=float))
print("written", outp)
