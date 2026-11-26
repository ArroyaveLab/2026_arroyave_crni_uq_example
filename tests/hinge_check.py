"""Check: value, violations and gradient of the hinge penalty on excluded phases (set D)."""
import sys, json, time, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); import numpy as np, crni
t = crni.GradTarget("setD_twophase_hinge")
for name, th in (("paper_opt", crni.PAPER_OPT), ("paper_start", crni.PAPER_START)):
    t0 = time.perf_counter(); p = t.parts(th); dt = time.perf_counter() - t0
    v = t.hinge.violations(th)
    print(name, {k: round(a, 2) for k, (a, g) in p.items()}, "s", round(dt, 1), "violations", len(v))
    import collections; print("  ", collections.Counter((z["phase"], round(z["T"] / 100) * 100) for z in v).most_common(8))
# gradient check of the hinge alone
th = crni.PAPER_OPT.copy(); P, G = t.hinge(th)
fd = []
for i in range(10):
    h = 1e-4 * max(1, abs(th[i])); a = th.copy(); b = th.copy(); a[i] += h; b[i] -= h
    fd.append((t.hinge(a)[0] - t.hinge(b)[0]) / (2 * h))
print("hinge P", P, "grad rel err", np.max(np.abs(np.array(fd) - G)) / (np.max(np.abs(G)) + 1e-30))
