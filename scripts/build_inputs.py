#!/usr/bin/env python3
"""Build the ESPEI dataset folders for the rebuild (data values never changed).

Treatment (Kunselman et al. 2026, Acta Mater. 315, 122412, section Cr-Ni):
  * a ZPF phase region with one known vertex composition and the other vertex
    unknown (None) keeps only the known vertex, i.e. it becomes a single-phase
    observation of that phase at that composition and temperature;
  * the one full tie-line (Zhang 2014, 1473 K) gets dataset weight 10
    (ESPEI ZPF std 1000/10 = 100 J/mol);
  * Jette 1934 is KEPT at dataset weight 0.25 (std 4000 J/mol; the dataset
    states no uncertainty) in set A, and left out in set B (the paper's choice).
Thermochemistry (dataset weight = ESPEI default std / sigma assigned here):
  * liquid HM_MIX: Saltykov 2002 and Thiedemann 1998 from ESPEI-datasets
    (original-source digitisations); Batalin 1983 and Sudavtsova 1999 from the
    Hao-Ruban-Xiong 2021 extraction (their Saltykov/Thiedemann copies are NOT
    used: same measurements, would double count). sigma 1500 J/mol (between-
    source scatter of 2-4 kJ/mol at x_Ni~0.6).
  * liquid ACR_CR (Hao extraction, 7 series): sigma 2500 J/mol in RT ln a
    (~15 % in activity at 1800 K, source-to-source spread).
  * BCC/FCC HM_FORM (Dench 1963, Watson 1995): sigma 1000 J/mol.
  * BCC ACR_CR (Pugliese 1970): sigma 1000 J/mol.
  * DFT HM_MIX (ORCHID; Liu et al., Calphad 52 (2016) 125, SQS, 0 K; A2 and
    gamma tables): ORCHID sigma_effective 7.2 kJ/mol-atom, entered at T=298.15 K
    (the excess enthalpy of an a+bT model does not depend on T).
  * NOT used: 'HM_MIX-BCC_A2-estimated' (an estimate, not an observation);
    ordered B2 / gamma' DFT rows (ordered phases are not in the model);
    Bracq 2021 FCC SQS value (temperature/context unspecified in ORCHID).
"""
import copy
import glob
import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
INP = RUN / "inputs"
CH = INP / "chapter-input-data"
HAO = INP / "hao-thermo"

ZPF_W = {"zhang2014impurity": 10.0, "jette1934xray": 0.25}
SIG = {"HM_MIX_LIQ": 1500.0, "ACR_LIQ": 2500.0, "HM_FORM_SOLID": 1000.0, "ACR_BCC": 1000.0,
       "DFT": 7200.0}
DEFAULT_STD = {"HM": 500.0, "ACR": 500.0}

DFT = {  # ORCHID records, kJ/mol-atom, x_Cr -> value
    "BCC_A2": [(0.75, 10.02, "ORC-ECXY5ARLCL"), (0.50, 9.15, "ORC-4AVGOLDRYY.862ddc"),
               (0.25, 4.98, "ORC-NNQH4SF765")],
    "FCC_A1": [(0.75, -12.69, "ORC-BJCC23KTCK.7c92d8"), (0.50, -9.03, "ORC-XYCE5GPHM6"),
               (0.25, -5.79, "ORC-CG6B7EHP2R.127230")],
}


def single_vertex(values):
    out, changed = [], 0
    for region in values:
        known = [v for v in region if v[2][0] is not None]
        if len(known) == len(region):
            out.append(region)
        else:
            out.append(known)
            changed += 1
    return out, changed


def build(dest: Path, keep_jette: bool):
    dest.mkdir(parents=True, exist_ok=True)
    log = {"zpf": {}, "thermo": {}}
    for f in sorted(CH.glob("zpf_*.json")):
        d = json.loads(f.read_text())
        ref = d["reference"]
        if ref == "jette1934xray" and not keep_jette:
            log["zpf"][ref] = "left out (set B)"
            continue
        d2 = copy.deepcopy(d)
        d2["values"], n = single_vertex(d["values"])
        # phases listed must equal the phases used by the (converted) values
        ph = sorted({v[0] for r in d2["values"] for v in r})  # ESPEI requires listed == used
        d2["phases"] = ph
        if ref in ZPF_W:
            d2["weight"] = ZPF_W[ref]
        d2["comment"] = d.get("comment", "") + (
            f" | REBUILD 2026-11-05: {n} region(s) with an unknown vertex kept as single-phase "
            f"observations of the known vertex (Kunselman 2026 treatment); weight {d2.get('weight', 1.0)}.")
        (dest / f.name).write_text(json.dumps(d2, indent=1))
        log["zpf"][ref] = {"points": len(d2["values"]), "converted_single": n, "weight": d2.get("weight", 1.0)}

    def put(d, name, sigma, default):
        d = copy.deepcopy(d)
        d["weight"] = default / sigma
        d["comment"] = d.get("comment", "") + f" | REBUILD: sigma {sigma} J/mol -> weight {d['weight']:.4g}."
        (dest / name).write_text(json.dumps(d, indent=1))
        n = len(d["values"][0][0]) if d["output"].startswith("HM") else len(d["values"][0][0])
        log["thermo"][name] = {"output": d["output"], "phases": d["phases"], "n": n, "sigma": sigma}

    for f in sorted(CH.glob("non-equilibrium-thermochemical_*.json")):
        d = json.loads(f.read_text())
        if d["reference"] == "estimated":
            log["thermo"][f.name] = "not used (estimate, not an observation)"
            continue
        sig = SIG["HM_MIX_LIQ"] if d["phases"] == ["LIQUID"] else SIG["HM_FORM_SOLID"]
        put(d, f.name, sig, DEFAULT_STD["HM"])
    for f in sorted(CH.glob("activity_*.json")):
        put(json.loads(f.read_text()), f.name, SIG["ACR_BCC"], DEFAULT_STD["ACR"])
    for f in sorted((HAO / "HM_MIX").glob("*.json")):
        if "saltykov" in f.name or "thiedemann" in f.name:
            log["thermo"]["hao/" + f.name] = "not used (duplicate of the ESPEI-datasets digitisation of the same measurements)"
            continue
        put(json.loads(f.read_text()), "hao_" + f.name, SIG["HM_MIX_LIQ"], DEFAULT_STD["HM"])
    for f in sorted((HAO / "ACR").glob("*.json")):
        d = json.loads(f.read_text())
        # same reference state written with the dependent component, as the chapter datasets do
        rc = d["reference_state"]["conditions"]
        if "X_NI" in rc and rc["X_NI"] == 0.0:
            rc.pop("X_NI")
            rc["X_CR"] = 1.0
        put(d, "hao_" + f.name, SIG["ACR_LIQ"], DEFAULT_STD["ACR"])
    for ph, rows in DFT.items():
        d = {"components": ["CR", "NI"], "phases": [ph],
             "solver": {"mode": "manual", "sublattice_site_ratios": [1],
                        "sublattice_configurations": [[["CR", "NI"]]] * len(rows),
                        "sublattice_occupancies": [[[x, round(1 - x, 6)]] for x, _, _ in rows]},
             "conditions": {"P": 101325, "T": 298.15}, "output": "HM_MIX",
             "values": [[[v * 1000.0 for _, v, _ in rows]]],
             "reference": "liu2016thermodynamic_dft (ORCHID)",
             "comment": "Liu, Lindwall, Gheno, Liu, Calphad 52 (2016) 125-142, SQS DFT at 0 K, tables 6 (A2) / 7 (gamma); "
                        "ORCHID records " + ",".join(r for _, _, r in rows) + "; kJ/mol-atom x1000."}
        put(d, f"dft_HM_MIX_{ph}_liu2016.json", SIG["DFT"], DEFAULT_STD["HM"])
    (dest.parent / "BUILD_LOG.json").write_text(json.dumps(log, indent=1))
    return log


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    for name, keep in (("setA_jette_w025", True), ("setB_no_jette", False)):
        lg = build(INP / "derived" / name / "datasets", keep)
        print(name, json.dumps(lg["zpf"]))
    print(json.dumps(lg["thermo"], indent=1))


# ------------------------------------------------------------------ set C (rule of 2026-11-05)
# Weights from the data's uncertainties, never hand-set. Observations that exist in ORCHID use the
# published ESPEI-bundle conversion (weight = ESPEI default sigma / sigma_effective_used):
#   liquid HM_MIX (Hao export, all four series incl. Saltykov, Thiedemann): sigma_eff 1000 J/mol -> 0.5
#     (the ESPEI-datasets Saltykov/Thiedemann files are the same measurements and take the same weight);
#   liquid ACR_Cr (Hao export): per-value sigma_mu ~2900 J/mol -> per-series mean of the per-value
#     weights (0.17240-0.17242; per-value spread < 1e-4 relative);
#   DFT HM_MIX (Liu 2016, ORCHID): sigma_eff 7.2 kJ/mol -> 500/7200;
#   ZPF: ORCHID leaves ZPF weight at 1 (driving-force tolerance), as ESPEI's default.
# Observations not in ORCHID (ESPEI-datasets ZPF incl. Jette 1934 and Zhang 2014, solid HM_FORM,
# Pugliese BCC ACR): ESPEI default sigma, weight 1. No x10 on Zhang, no 0.25 on Jette.
# The paper's vertex treatment (unknown vertex dropped) is kept, so only the weighting differs from A.
ORCHID_ACR_W = {"kat1733": 0.17242, "kat1783": 0.17240, "sud1833": 0.172406, "kat1833": 0.172419,
                "gilby": 0.172419, "pei": 0.17240, "sud1873": 0.172407}
# Optional: a folder with an ORCHID export of the same activity series (set CRNI_ORCHID_EXPORT).
# It is used only to record the weights of that export in the build log for comparison.
EXPORT = Path(os.environ["CRNI_ORCHID_EXPORT"]) if os.environ.get("CRNI_ORCHID_EXPORT") else None


def build_c(dest: Path):
    import glob as _g
    dest.mkdir(parents=True, exist_ok=True)
    log = {"rule": "uncertainty-derived (ORCHID sigma_effective where the observation is in ORCHID, else ESPEI default)",
           "datasets": {}}
    # check the ACR series weights against the export files (same series order, same T and n)
    exp = {}
    for f in (sorted(_g.glob(str(EXPORT / "*acr_cr-*.json"))) if EXPORT else []):
        d = json.loads(Path(f).read_text())
        exp.setdefault(f.split("acr_cr-")[1][:2], []).append(d["weight"])

    def put(d, name, w, why):
        d = copy.deepcopy(d)
        d.pop("weight", None)
        if w != 1.0:
            d["weight"] = w
        d["comment"] = d.get("comment", "") + f" | REBUILD set C: weight {w:.6g} ({why})."
        (dest / name).write_text(json.dumps(d, indent=1))
        log["datasets"][name] = {"output": d["output"], "phases": d["phases"], "weight": w, "basis": why}

    for f in sorted(CH.glob("zpf_*.json")):
        d = json.loads(f.read_text())
        d["values"], n = single_vertex(d["values"])
        d["phases"] = sorted({v[0] for r in d["values"] for v in r})
        put(d, f.name, 1.0, "ZPF, ESPEI default 1000 J/mol driving force (ORCHID leaves ZPF at 1)")
    for f in sorted(CH.glob("non-equilibrium-thermochemical_*.json")):
        d = json.loads(f.read_text())
        if d["reference"] == "estimated":
            continue
        if d["phases"] == ["LIQUID"]:
            put(d, f.name, 0.5, "same measurement as ORCHID Hao record, sigma_effective 1000 J/mol")
        else:
            put(d, f.name, 1.0, "not in ORCHID, ESPEI default 500 J/mol")
    for f in sorted(CH.glob("activity_*.json")):
        put(json.loads(f.read_text()), f.name, 1.0, "not in ORCHID, ESPEI default 500 J/mol")
    for f in sorted((HAO / "HM_MIX").glob("*.json")):
        if "saltykov" in f.name or "thiedemann" in f.name:
            continue
        put(json.loads(f.read_text()), "hao_" + f.name, 0.5, "ORCHID Hao record, sigma_effective 1000 J/mol")
    for f in sorted((HAO / "ACR").glob("*.json")):
        d = json.loads(f.read_text())
        rc = d["reference_state"]["conditions"]
        if "X_NI" in rc and rc["X_NI"] == 0.0:
            rc.pop("X_NI"); rc["X_CR"] = 1.0
        key = f.name.split("ACR_CR-LIQUID-")[1].split("-")[0]
        put(d, "hao_" + f.name, ORCHID_ACR_W[key], "ORCHID Hao records, sigma_mu ~2900 J/mol, mean of per-value weights")
    for ph, rows in DFT.items():
        src = json.loads((INP / "derived" / "setA_jette_w025" / "datasets" / f"dft_HM_MIX_{ph}_liu2016.json").read_text())
        put(src, f"dft_HM_MIX_{ph}_liu2016.json", 500.0 / 7200.0, "ORCHID DFT sigma_effective 7.2 kJ/mol")
    log["export_acr_weights_by_series"] = {k: [min(v), max(v), len(v)] for k, v in exp.items()}
    (dest.parent / "BUILD_LOG.json").write_text(json.dumps(log, indent=1))
    return log


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "C":
    lg = build_c(INP / "derived" / "setC_uncertainty" / "datasets")
    for k, v in lg["datasets"].items():
        print(f"{v['weight']:.5g}  {v['output']:8s} {k[:70]}")
