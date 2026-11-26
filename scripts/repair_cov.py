"""Init-ball covariance for mcmc: curvature.py's -H, with the 3 softest eigenvalues replaced by
line-scan curvatures kappa_k = -(d(+1)+d(-1)) along those eigenvectors (soft_scan.py), because the
finite-difference Hessian does not resolve curvatures below ~0.1 (its step-size disagreement is ~6%
of max|H| ~ 32). Only the initial walker ball uses this; posterior and analysis rules are unchanged.
usage: repair_cov.py <curvature_json> <soft_scan_json> <out_json>"""
import json, sys
import numpy as np
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__))); import crni
c = json.load(open(sys.argv[1])); s = json.load(open(sys.argv[2]))
H = np.array(c["H_q"]); w, V = np.linalg.eigh(-H); w2 = w.copy()
for k in range(3):
    r = {a: b for a, b, _ in s[f"dir{k}"]}
    w2[k] = -(r[1] + r[-1])
assert np.all(w2 > 0), w2
cov_q = V @ np.diag(1 / w2) @ V.T; D = np.diag(crni.SCALE); cov_t = D @ cov_q @ D
out = dict(c); out.update({"cov_theta": cov_t.tolist(), "sd_theta": np.sqrt(np.diag(cov_t)).tolist(),
           "eig_negH_q_fd": w.tolist(), "eig_negH_q_used": w2.tolist(), "positive_definite_used": True,
           "init_cov_note": __doc__})
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({"eig_fd": w[:4].tolist(), "eig_used": w2[:4].tolist(), "sd_theta": out["sd_theta"]}))
