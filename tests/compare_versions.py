"""Check: likelihood terms from ESPEI 0.9.0 and from the analytic-gradient ESPEI agree.
usage: compare_versions.py <scripts_dir> <grad|e09> <setname>"""
import sys, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1]); import numpy as np, crni
which = sys.argv[2]; setname = sys.argv[3]
out = {}
for name, th in (("paper_start", crni.PAPER_START), ("paper_opt", crni.PAPER_OPT)):
    if which == "grad":
        t = crni.GradTarget(setname) if not out else t
        t0 = time.perf_counter(); p = t.parts(th); dt = time.perf_counter() - t0
        out[name] = {"parts": {k: v for k, (v, g) in p.items()}, "grad_finite": {k: bool(np.all(np.isfinite(g))) for k, (v, g) in p.items()}, "s": dt}
    else:
        f = crni.espei09_lnprob(setname) if not out else f
        t0 = time.perf_counter(); p = f.parts(th); dt = time.perf_counter() - t0
        out[name] = {"parts": p, "s": dt}
print(json.dumps(out, indent=1))
