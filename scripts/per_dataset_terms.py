#!/usr/bin/env python3
"""Per-dataset log-likelihood terms (ESPEI-grad value path) and hinge per source, at several parameter sets.
usage: per_dataset_terms.py <setname> <out_json> name=theta.json|paper_opt ...
"""
import json, sys, warnings, collections
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, crni
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from pycalphad import Database

setname, outp = sys.argv[1], Path(sys.argv[2])
pts = {}
for a in sys.argv[3:]:
    n, v = a.split("=")
    pts[n] = crni.PAPER_OPT if v == "paper_opt" else np.asarray(json.loads(Path(v).read_text())["theta"])
res = {n: {} for n in pts}
for f in sorted(crni.datasets_dir(setname).glob("*.json")):
    ds = load_datasets([str(f)])
    ctx = setup_context(Database(str(crni.TDB)), ds, symbols_to_fit=crni.NAMES, phase_models=crni.phase_models())
    for n, th in pts.items():
        res[n][f.name] = float(sum(float(r.get_likelihood(th)) for r in ctx["residual_objs"]))
    print(f.name, {n: round(res[n][f.name], 2) for n in pts}, flush=True)
side = crni.INP / "derived" / setname / "hinge" / "CR-NI" / "cr-ni-model.json"
if side.exists():
    from phaseforgeplus.utils.hinge import HingeOptions, HingePenalty
    sc = json.loads(side.read_text())
    keys = ("T", "x", "expected", "sigma", "tie_line", "confidence", "penalize")
    hp = HingePenalty(Database(str(crni.TDB)), crni.NAMES, [{k: c[k] for k in keys if k in c} for c in sc["cells"]],
                      "NI", HingeOptions(margin=sc["margin_J_per_mol_atom"]))
    from phaseforgeplus.utils.hinge import _hinge_term
    for n, th in pts.items():
        by = collections.defaultdict(float)
        for k, (cell, pen) in enumerate(zip(hp.cells, hp.penalized)):
            for ph in pen:
                df = hp._driving_force(k, ph, th)[0]
                by["hinge:" + sc["cells"][k]["orchid"]["dataset"]] -= _hinge_term(df, hp.options.margin, hp._sigma(cell, ph), cell.confidence)[0]
        res[n].update(by)
for n, th in pts.items():
    res[n]["prior"] = crni.log_prior(th)
outp.write_text(json.dumps(res, indent=1))
