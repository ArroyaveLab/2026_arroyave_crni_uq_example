#!/usr/bin/env python3
"""Step 3: curvature of the log posterior at the MAP (ESPEI-grad analytic gradient, central differences).

H_ij = d/dq_j g_i, by central differences of the analytic gradient with two step sizes
(relative 1e-3 and 3e-3 of max(1,|q|)); symmetrised. Laplace covariance = inv(-H) in q, mapped to
theta with SCALE. Also reports eigenvalues, correlations, Laplace sd, and the asymmetry and
step-size disagreement of H. usage: curvature.py <setname> [theta.json]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni


def hessian(t, q, rel):
    n = len(q)
    H = np.zeros((n, n))
    bad = []
    for j in range(n):
        h = rel * max(1.0, abs(q[j]))
        qp, qm = q.copy(), q.copy()
        qp[j] += h; qm[j] -= h
        vp, gp = t(qp); vm, gm = t(qm)
        if not (np.isfinite(vp) and np.isfinite(vm) and np.all(np.isfinite(gp)) and np.all(np.isfinite(gm))):
            bad.append({"j": j, "vp": float(vp), "vm": float(vm)})
        H[:, j] = (gp - gm) / (2 * h)
    return H, bad


def main():
    setname = sys.argv[1]
    if len(sys.argv) > 2:
        theta = np.asarray(json.loads(Path(sys.argv[2]).read_text())["theta"])
    else:
        ms = json.loads((crni.RUN / "fit" / f"multistart_{setname}.json").read_text())
        theta = np.asarray(ms[0]["theta"])
    t = crni.GradTarget(setname)
    q = theta / crni.SCALE
    v0, g0 = t(q)
    H1, b1 = hessian(t, q, 1e-3)
    H2, b2 = hessian(t, q, 3e-3)
    asym = float(np.max(np.abs(H1 - H1.T)) / np.max(np.abs(H1)))
    stepdiff = float(np.max(np.abs(H1 - H2)) / np.max(np.abs(H1)))
    H = 0.5 * (H1 + H1.T)
    w, V = np.linalg.eigh(-H)
    out = {"set": setname, "theta_map": theta.tolist(), "logpost": float(v0), "grad_norm_q": float(np.linalg.norm(g0)),
           "H_q": H.tolist(), "eig_negH_q": w.tolist(), "rel_asymmetry": asym, "rel_stepsize_diff": stepdiff,
           "nonfinite": b1 + b2}
    if np.all(w > 0):
        cov_q = V @ np.diag(1 / w) @ V.T
        out["positive_definite"] = True
    else:
        wc = np.clip(w, np.max(w) * 1e-10, None)
        cov_q = V @ np.diag(1 / wc) @ V.T
        out["positive_definite"] = False
    D = np.diag(crni.SCALE)
    cov_t = D @ cov_q @ D
    sd = np.sqrt(np.diag(cov_t))
    out["cov_theta"] = cov_t.tolist()
    out["sd_theta"] = sd.tolist()
    out["corr"] = (cov_t / np.outer(sd, sd)).tolist()
    out["condition_number_q"] = float(w.max() / max(w.min(), 1e-300))
    out["stiffest_direction_q"] = V[:, -1].tolist()
    out["softest_direction_q"] = V[:, 0].tolist()
    (crni.RUN / "fit" / f"curvature_{setname}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("set", "logpost", "grad_norm_q", "eig_negH_q", "rel_asymmetry",
                                          "rel_stepsize_diff", "positive_definite", "condition_number_q")}))
    for lab, th, s in zip(crni.LABELS, theta, sd):
        print(f"{lab:9s} {th:12.3f} +- {s:10.3f}")


if __name__ == "__main__":
    main()
