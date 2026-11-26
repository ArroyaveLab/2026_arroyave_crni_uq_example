"""lnprob along the softest eigenvectors of -H at a mode (q coords). usage: soft_scan.py <scripts> <set> <curvature_json>"""
import json, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, sys.argv[1]); import crni
c = json.load(open(sys.argv[3])); H = np.array(c["H_q"]); th = np.array(c["theta_map"]); q = th / crni.SCALE
w, V = np.linalg.eigh(-H)
t = crni.GradTarget(sys.argv[2]); v0 = float(t(q)[0])
out = {"logpost0": v0, "eig": w[:3].tolist()}
for k in range(3):
    row = []
    for s in (-3, -1, -0.3, -0.1, 0.1, 0.3, 1, 3):
        v, g = t(q + s * V[:, k]); row.append([s, float(v) - v0, float(g @ V[:, k])])
    out[f"dir{k}"] = row; out[f"v{k}_theta"] = (V[:, k] * crni.SCALE).tolist()
print(json.dumps(out))
