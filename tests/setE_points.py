"""Per-observation likelihood differences between the two set E fitted points (activities and ZPF)."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import numpy as np, crni
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from pycalphad import Database
th = {n: np.asarray(json.load(open(f"fit/setE_{f}_theta.json"))["theta"]) for n, f in (("correct", "correct_best"), ("gap", "gapmode_best"))}
out = {}
for fn in ["activity_CR-NI-ACR_CR-BCC_A2-pugliese1970activities.json", "activity_CR-NI-ACR_CR-BCC_A2-pugliese1970activities-0.json",
           "activity_CR-NI-ACR_CR-BCC_A2-pugliese1970activities-1.json", "activity_CR-NI-ACR_CR-BCC_A2-pugliese1970activities-2.json",
           "activity_CR-NI-ACR_CR-BCC_A2-pugliese1970activities-3.json", "zpf_CR-NI-ZPF-BCC_A2-FCC_A1-collins1988electron.json",
           "zpf_CR-NI-ZPF-BCC_A2-FCC_A1-bechtoldt1961redetermination.json"]:
    f = crni.datasets_dir("setE_orchid_sigmas") / fn; d = json.load(open(f))
    ctx = setup_context(Database(str(crni.TDB)), load_datasets([str(f)]), symbols_to_fit=crni.NAMES, phase_models=crni.phase_models())
    r = ctx["residual_objs"][0] if type(ctx["residual_objs"][0]).__name__ != "EquilibriumPropertyResidual" else None
    r = [o for o in ctx["residual_objs"] if type(o).__name__ in ("ActivityResidual", "ZPFResidual")][0]
    c = {}
    for n, t in th.items():
        res, w = r.get_residuals(t); res, w = np.asarray(res, float), np.asarray(w, float)
        sc = 1000.0 / w if type(r).__name__ == "ZPFResidual" else w
        c[n] = 0.5 * (res / sc) ** 2
    gain = c["correct"] - c["gap"]
    if "activity" in fn:
        xs = d["conditions"]["X_CR"]; lab = [f"T={d['conditions']['T']} x_Cr={x}" for x in xs]
    else:
        lab = []
        for T, reg in zip(d["conditions"]["T"], d["values"]):
            for v in reg:
                lab.append(f"T={T} {v[0]} x={v[2][0]}")
    o = np.argsort(-gain)[:4]
    out[fn] = [(lab[k] if k < len(lab) else str(k), round(float(gain[k]), 3)) for k in o]
    print(fn[:60], "sum", round(float(gain.sum()), 2), out[fn])
json.dump(out, open("results/setE_gap_gain_points.json", "w"), indent=1)
