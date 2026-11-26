"""Diagnose the parameter vector at which the set A fit returned a non-finite value.
usage: inf_point.py <grad|e09>"""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); import numpy as np, crni
which = sys.argv[1]
r = json.load(open("fit/multistart_setA_jette_w025.json"))[0]
q = np.array(r["nonfinite_events"][0]["q"]); th = q * crni.SCALE
if which == "grad":
    t = crni.GradTarget("setA_jette_w025")
    print("grad-version value-only:", {type(o).__name__: float(o.get_likelihood(th)) for o in t.res})
    print("theta_end value-only:", {type(o).__name__: float(o.get_likelihood(np.array(r['theta']))) for o in t.res})
else:
    f = crni.espei09_lnprob("setA_jette_w025")
    print("espei 0.9.0:", f.parts(th))
