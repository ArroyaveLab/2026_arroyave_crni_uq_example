#!/usr/bin/env python3
"""Set D (2026-11-05): set C's uncertainty-derived weights, with two changes in data treatment.

1. ZPF points keep every vertex as printed: a two-phase point with one reported composition stays two-phase,
   the other phase present with composition null (ESPEI-datasets files unchanged, weight 1).
2. Excluded phases: at each existing observation, the phases stable elsewhere in Cr-Ni (liquid,
   bcc, fcc in the sources' diagrams; CrNi2 is not in this model) but not at that state go to the
   PhaseForge+ hinge (phaseforgeplus.utils.hinge), margin tau = 100 J/mol-atom. No source states an
   exclusion, so every cell is 'assumed from a diagram' (option C): reduced confidence CONF_C.
   Cells come only from existing observations:
     ZPF point            expected = known-vertex phase at its x (Zhang: both ends, tie_line);
                          penalize = model phases not in the observation
     liquid HM_MIX / ACR  expected LIQUID at x (liquid alloys measured);  penalize BCC_A2, FCC_A1
     Dench/Watson HM_FORM expected the measured phase (two-phase points were neglected by the
                          dataset author);  penalize the other two
   Not used: Pugliese activities (state phases not given), DFT (0 K calculations).
Sidecar: inputs/derived/setD_twophase_hinge/hinge/CR-NI/cr-ni-model.json (orchid-data-next schema branch
format: {system, tdb, tdb_sha256, element, cells[]}; each cell's `orchid` key holds the provenance and is
stripped before PhaseForge+ reads the cell).
"""
import copy
import hashlib
import json
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
INP = RUN / "inputs"
CH = INP / "chapter-input-data"
SETC = INP / "derived" / "setC_uncertainty" / "datasets"
DEST = INP / "derived" / "setD_twophase_hinge"
SIGMA_HINGE = 1000.0   # J/mol-atom, ESPEI's ZPF driving-force scale
CONF_C = 0.5           # option C, reduced confidence for exclusions assumed from a diagram (provisional)
LIQ_WINDOW = 0.02      # x_Ni window for 'liquidus observation at that composition' (about 2 x ORCHID ZPF sigma_x 0.011)
TAU = 100.0            # margin = state tolerance, J/mol-atom
MODEL = ["BCC_A2", "FCC_A1", "LIQUID"]


def main():
    ds = DEST / "datasets"
    ds.mkdir(parents=True, exist_ok=True)
    log = {"datasets": {}, "hinge": {}}
    cells = []
    for f in sorted(CH.glob("zpf_*.json")):
        d = json.loads(f.read_text())
        d2 = copy.deepcopy(d)
        d2.pop("weight", None)
        d2["phases"] = sorted({v[0] for r in d["values"] for v in r})
        d2["comment"] = d.get("comment", "") + (" | REBUILD set D, observation (ii): values as printed, null vertex kept "
                                                 "(ESPEI zero-phase-fraction condition of the partner phase), weight 1.")
        (ds / f.name).write_text(json.dumps(d2, indent=1))
        log["datasets"][f.name] = {"output": "ZPF", "weight": 1.0, "n": len(d["values"]), "role": "(ii) two-phase, null vertex"}
        # (i) the known phase at its composition, for the regions with one known end (same parent point)
        idx1 = [i for i, r in enumerate(d["values"]) if 0 < sum(v[2][0] is not None for v in r) < len(r)]
        if idx1:
            Tl = d["conditions"]["T"] if isinstance(d["conditions"]["T"], list) else [d["conditions"]["T"]] * len(d["values"])
            d1 = copy.deepcopy(d)
            d1.pop("weight", None)
            d1["values"] = [[v for v in d["values"][i] if v[2][0] is not None] for i in idx1]
            d1["conditions"] = dict(d["conditions"], T=[Tl[i] for i in idx1])
            d1["phases"] = sorted({v[0] for r in d1["values"] for v in r})
            d1["comment"] = d.get("comment", "") + (" | REBUILD set D, observation (i): the known phase at its printed "
                                                    "composition (single-phase), derived from the same parent points as "
                                                    "the (ii) file; parent indices " + json.dumps(idx1) + "; weight 1.")
            name1 = f.name.replace("zpf_", "zpf_single_")
            (ds / name1).write_text(json.dumps(d1, indent=1))
            log["datasets"][name1] = {"output": "ZPF", "weight": 1.0, "n": len(idx1), "role": "(i) single-phase, same parents",
                                      "parent_dataset": f.name, "parent_indices": idx1}
        Ts = d["conditions"]["T"]
        Ts = Ts if isinstance(Ts, list) else [Ts] * len(d["values"])
        for i, (t, reg) in enumerate(zip(Ts, d["values"])):
            phases = [v[0] for v in reg]
            known = [(v[0], v[2][0] if v[1] == ["NI"] else 1 - v[2][0]) for v in reg if v[2][0] is not None]
            pen = [p for p in MODEL if p not in phases]
            if not pen:
                continue
            src = {"dataset": f.name, "index": i, "reference": d["reference"], "relation": "stable_elsewhere",
                   "basis": "assumed_from_diagram (option C)"}
            if len(known) == 2:
                cells.append({"T": float(t), "x": float(sum(x for _, x in known) / 2), "expected": [p for p, _ in known],
                              "tie_line": {p: float(x) for p, x in known}, "penalize": pen, "sigma": SIGMA_HINGE,
                              "confidence": CONF_C, "orchid": src})
            elif len(known) == 1:
                p, x = known[0]
                cells.append({"T": float(t), "x": float(x), "expected": [p], "penalize": pen, "sigma": SIGMA_HINGE,
                              "confidence": CONF_C, "orchid": src})
    # experimental liquidus observations: ZPF regions with a known LIQUID vertex (x_Ni, T)
    liq_obs = []
    for f in sorted(CH.glob("zpf_*.json")):
        d = json.loads(f.read_text())
        Tl = d["conditions"]["T"] if isinstance(d["conditions"]["T"], list) else [d["conditions"]["T"]] * len(d["values"])
        for t, reg in zip(Tl, d["values"]):
            for v in reg:
                if v[0] == "LIQUID" and v[2][0] is not None:
                    liq_obs.append((v[2][0] if v[1] == ["NI"] else 1 - v[2][0], float(t)))
    no_excl = []
    for f in sorted(SETC.glob("*.json")):
        if f.name.startswith("zpf_"):
            continue
        d = json.loads(f.read_text())
        d["comment"] = d.get("comment", "").replace("REBUILD set C", "REBUILD set C/D")
        (ds / f.name).write_text(json.dumps(d, indent=1))
        log["datasets"][f.name] = {"output": d["output"], "weight": d.get("weight", 1.0)}
        src = {"dataset": f.name, "reference": d["reference"][:120], "relation": "stable_elsewhere",
               "basis": "assumed_from_diagram (option C)"}
        if f.name.startswith("dft_") or f.name.startswith("activity_CR-NI-ACR_CR-BCC_A2-pugliese"):
            continue
        T = float(d["conditions"]["T"])
        if d["output"].startswith("HM"):
            xs = [occ[0][1] for occ in d["solver"]["sublattice_occupancies"]]  # (CR, NI) -> x_Ni
        else:
            c = d["conditions"]
            xs = c["X_NI"] if "X_NI" in c else [1 - v for v in c["X_CR"]]
        ph = d["phases"][0]
        pen = [p for p in MODEL if p != ph]
        for i, x in enumerate(xs):
            if not 0.0 < float(x) < 1.0:
                continue
            if ph == "LIQUID":  # exclusions only above every experimental liquidus observation near x
                near = [t for xl, t in liq_obs if abs(xl - float(x)) <= LIQ_WINDOW]
                if not near or T <= max(near):
                    no_excl.append({"dataset": f.name, "index": i, "T": T, "x_Ni": float(x),
                                    "reason": "no liquidus observation within 0.02" if not near
                                    else f"not above liquidus observation(s) up to {max(near):.0f} K"})
                    continue
            cells.append({"T": T, "x": float(x), "expected": [ph], "penalize": pen, "sigma": SIGMA_HINGE,
                          "confidence": CONF_C, "orchid": dict(src, index=i)})
    tdb = INP / "cr-ni-model.tdb"
    side = {"system": "CR-NI", "tdb": "cr-ni-model.tdb", "tdb_sha256": hashlib.sha256(tdb.read_bytes()).hexdigest(),
            "element": "NI", "margin_J_per_mol_atom": TAU, "sigma_J_per_mol_atom": SIGMA_HINGE,
            "confidence_option_C": CONF_C, "cells": cells}
    hd = DEST / "hinge" / "CR-NI"
    hd.mkdir(parents=True, exist_ok=True)
    (hd / "cr-ni-model.json").write_text(json.dumps(side, indent=1))
    import collections
    log["hinge"] = {"n_cells": len(cells), "margin": TAU, "sigma": SIGMA_HINGE, "confidence_option_C": CONF_C,
                    "by_expected_and_penalized": dict(collections.Counter(
                        f"{'+'.join(c['expected'])} | not {'+'.join(c['penalize'])}" for c in cells))}
    log["hinge"]["liquid_states_without_exclusion"] = no_excl
    log["hinge"]["n_liquidus_observations"] = len(liq_obs)
    log["note"] = ("ZPF regions with one known end enter twice, as (i) the known phase single-phase and (ii) the two-phase "
                   "null-vertex ZPF condition, both derived from the same parent point (rule of 2026-11-05); no further "
                   "double counting is claimed or corrected. Hinge confidence 0.5 is an assumed value.")
    (DEST / "BUILD_LOG.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log["hinge"], indent=1))


if __name__ == "__main__":
    main()
