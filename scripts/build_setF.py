#!/usr/bin/env python3
"""Set F = set E + BCC convexity term at observed single-phase BCC states (2026-11-05) + Jette duplicates once.

Convexity (scripts/bcc_convexity.py): at every observed single-phase BCC state (the BCC known vertices of the ZPF data,
set E (i) files, and the BCC end of Zhang's tie-line), one-sided penalty on G''_BCC(x,T) < 0:
P = 0.5 * sum (E_def / sigma)^2, E_def = -G'' * w^2 / 8 (sagitta of a chord of width w = 0.1), sigma = 1000 J/mol-atom.
Nowhere else (a metastable gap outside the observed domain stays allowed). No BCC miscibility gap is reported in the
sources we hold (Hao 2021, Lee, Pugliese 1970, Chatterji-Kaufman 1971, ORCHID records).
Jette 1934: the ESPEI-datasets file holds 6 identical row pairs (T, phases, compositions); in the derived set F files
each is counted once (data file inputs/chapter-input-data unchanged).
"""
import json
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
E = RUN / "inputs" / "derived" / "setE_orchid_sigmas"
F = RUN / "inputs" / "derived" / "setF_convexity"


def dedupe(d):
    Tl = d["conditions"]["T"]
    seen, keep, dropped = set(), [], []
    for i, (t, r) in enumerate(zip(Tl, d["values"])):
        k = (t, json.dumps(r))
        if k in seen:
            dropped.append(i)
            continue
        seen.add(k); keep.append(i)
    d = dict(d)
    d["conditions"] = dict(d["conditions"], T=[Tl[i] for i in keep])
    d["values"] = [d["values"][i] for i in keep]
    return d, dropped


def main():
    (F / "datasets").mkdir(parents=True, exist_ok=True)
    (F / "hinge" / "CR-NI").mkdir(parents=True, exist_ok=True)
    shutil.copy2(E / "hinge" / "CR-NI" / "cr-ni-model.json", F / "hinge" / "CR-NI" / "cr-ni-model.json")
    log = {"base": "setE_orchid_sigmas (weights, hinge unchanged)", "jette_duplicates_dropped": {}}
    states = []
    for f in sorted((E / "datasets").glob("*.json")):
        d = json.loads(f.read_text())
        if "jette1934" in f.name:
            d, dropped = dedupe(d)
            d["comment"] = d.get("comment", "") + f" | REBUILD set F: identical rows counted once (dropped indices {dropped})."
            log["jette_duplicates_dropped"][f.name] = dropped
        (F / "datasets" / f.name).write_text(json.dumps(d, indent=1))
        if f.name.startswith("zpf_single_") or "zhang2014" in f.name:
            for T, r in zip(d["conditions"]["T"], d["values"]):
                for v in r:
                    if v[0] == "BCC_A2" and v[2][0] is not None:
                        states.append([float(T), v[2][0] if v[1] == ["NI"] else 1 - v[2][0], f.name])
    uniq = sorted({(T, round(x, 6)): s for T, x, s in states}.items())
    conv = {"w_mole_fraction": 0.1, "sigma_J_per_mol_atom": 1000.0,
            "rule": "one-sided on G''_BCC < 0 at observed single-phase BCC states only",
            "states": [[T, x, s] for (T, x), s in uniq]}
    (F / "convexity.json").write_text(json.dumps(conv, indent=1))
    log["convexity_states"] = len(conv["states"])
    (F / "BUILD_LOG.json").write_text(json.dumps(log, indent=1))
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
