"""Diagnose the activity observation that returned a non-finite residual during the set A fit."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); import numpy as np, crni
from espei.error_functions.activity_error import calculate_activity_residuals
r = json.load(open("fit/multistart_setA_jette_w025.json"))[0]
th = np.array(r["nonfinite_events"][0]["q"]) * crni.SCALE
f = crni.espei09_lnprob("setA_jette_w025")
act = [o for o in f.ctx["residual_objs"] if type(o).__name__ == "ActivityResidual"][0]
res, w = act.get_residuals(th)
res = np.asarray(res); print(len(res), "nonfinite idx", np.where(~np.isfinite(res))[0], res[~np.isfinite(res)])
# map indices to datasets
from espei.datasets import load_datasets, recursive_glob
from tinydb import where
ds = act._activity_likelihood_kwargs["datasets"].search((where("output").test(lambda x: "ACR" in x)) & (where("components").test(lambda x: set(x).issubset({"CR","NI","VA"}))))
i = 0
for d in ds:
    n = len(np.atleast_1d(d["conditions"].get("X_CR", d["conditions"].get("X_NI"))))
    bad = [k for k in range(i, i+n) if not np.isfinite(res[k])]
    if bad: print(d["reference"][:80], d["conditions"]["T"], [np.atleast_1d(d["conditions"].get("X_CR", d["conditions"].get("X_NI")))[k-i] for k in bad])
    i += n
print("total", i)
