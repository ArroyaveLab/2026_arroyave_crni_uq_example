#!/usr/bin/env python3
"""Evaluate parameter ensembles once and store the precomputed response tensor (chapter Section 7).

Ensembles
  joint        N_J draws from the pooled post-burn-in samples of both runs, thinned evenly (propagate.py rule);
               (run, step, walker) kept for every draw. The first check rebuilds the 400-draw ensemble of
               results/notconverged/setE_prop400_joint_draws.npy and requires it to be identical.
  independent  the joint draws with every parameter column permuted independently (seed 20261106)
  laplace      N_L draws from N(theta_opt, Sigma_Laplace), Sigma_Laplace = the curvature at the accepted optimum
               (cr-ni-setH-20261106/inputs/setE_spark/curvature_setE_run1_b557.json, cov_theta: finite-difference
               Hessian of the ESPEI-grad log posterior, 3 softest eigenvalues replaced by line-scan curvatures)
Per draw: class (ok/gap/other), BCC/(BCC+FCC) and (BCC+FCC)/FCC boundaries on TT, region at (0.52, 1473 K),
single-phase-FCC indicator on the decision map (MAP_T x MAP_X), and candidate-measurement responses
(boundary compositions; RT ln a_Cr from the stable hull, reference pure bcc Cr).
usage: compute_ensembles.py <N_J> <N_L> <nproc>  -> data/ensembles.npz, data/ensembles_meta.json
"""
import json
import sys
import time
from multiprocessing import get_context

import numpy as np

import common as C

CURV = C.RUN / "inputs/setE_spark/curvature_setE_run1_b557.json"
T_BD = [1300.0, 1350.0, 1400.0, 1450.0, 1500.0, 1550.0]
T_ACT = [1300.0, 1400.0, 1500.0]
X_ACT_BCC = [0.10, 0.20, 0.30]
X_ACT_FCC = [0.60, 0.70, 0.80]
CANDS = ([("x_bcc_boundary", t, None) for t in T_BD] + [("x_fcc_boundary", t, None) for t in T_BD]
         + [("mu_Cr_bcc_field", t, x) for t in T_ACT for x in X_ACT_BCC]
         + [("mu_Cr_fcc_field", t, x) for t in T_ACT for x in X_ACT_FCC])
G = {}


def _init():
    G["tt"] = C.ref_grid(C.TT)
    G["map"] = C.ref_grid(C.MAP_T)
    G["dec"] = C.ref_grid([C.T_DEC])
    G["act"] = C.ref_grid(T_ACT)


def evaluate(th):
    cls, top = C.classify(th, G["tt"])
    b = C.pdfast.boundaries(th, C.TT, G["tt"])
    reg = C.pdfast.phase_at(th, C.T_DEC, C.X_DEC, G["dec"])
    fm = C.fcc_map(th, G["map"])
    resp = []
    for kind, t, x in CANDS:
        if kind == "x_bcc_boundary":
            resp.append(b["bcc_bf"][int(np.argmin(abs(C.TT - t)))])
        elif kind == "x_fcc_boundary":
            resp.append(b["fcc_bf"][int(np.argmin(abs(C.TT - t)))])
        else:
            resp.append(C.mu_cr_rel(th, t, x, G["act"][:, T_ACT.index(t), :]))
    return C.CLASSES.index(cls), b["bcc_bf"], b["fcc_bf"], reg, fm, np.array(resp), top["T_liquid_min"]


def run(pool, D):
    res = pool.map(evaluate, list(D), chunksize=8)
    return {"cls": np.array([r[0] for r in res], np.int8), "bcc_bf": np.array([r[1] for r in res]),
            "fcc_bf": np.array([r[2] for r in res]), "region": np.array([r[3] for r in res]),
            "fccmap": np.array([r[4] for r in res]), "resp": np.array([r[5] for r in res]),
            "tliq": np.array([r[6] for r in res])}


def main():
    nj, nl, nproc = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    j400, _ = C.reconstruct_draws(400)
    saved = np.load(C.RUN / "results/notconverged/setE_prop400_joint_draws.npy")
    same400 = bool(np.array_equal(j400, saved))
    print("rebuild of the 400-draw joint ensemble identical:", same400, flush=True)
    joint, meta = C.reconstruct_draws(nj)
    indep = C.independent_of(joint, 20261106)
    cur = json.loads(CURV.read_text())
    th0 = np.asarray(cur["theta_map"])
    assert np.allclose(th0, C.theta_of("accepted"))
    rng = np.random.default_rng(20261107)
    lap = rng.multivariate_normal(th0, np.asarray(cur["cov_theta"]), size=nl)
    out = {"joint_theta": joint, "joint_meta": meta, "indep_theta": indep, "laplace_theta": lap}
    t0 = time.time()
    with get_context("fork").Pool(nproc, initializer=_init) as pool:
        for name, D in (("joint", joint), ("indep", indep), ("laplace", lap), ("j400", j400)):
            r = run(pool, D)
            for k, v in r.items():
                out[f"{name}_{k}"] = v
            print(name, len(D), "draws", round(time.time() - t0), "s; class fractions",
                  np.bincount(r["cls"], minlength=3) / len(D), flush=True)
    np.savez(C.DATA / "ensembles.npz", TT=C.TT, MAP_T=C.MAP_T, MAP_X=C.MAP_X, **out)
    meta_out = {"N_joint": nj, "N_laplace": nl, "rebuild_400_identical": same400, "seed_indep": 20261106,
                "seed_laplace": 20261107, "curvature_file": str(CURV.relative_to(C.RUN)),
                "laplace_note": cur.get("init_cov_note", ""), "eig_negH_q_fd": cur.get("eig_negH_q_fd"),
                "eig_negH_q_used": cur.get("eig_negH_q_used"),
                "candidates": [{"i": i, "kind": k, "T": t, "x_Ni": x} for i, (k, t, x) in enumerate(CANDS)],
                "wall_s": time.time() - t0, "nproc": nproc}
    (C.DATA / "ensembles_meta.json").write_text(json.dumps(meta_out, indent=1))


if __name__ == "__main__":
    main()
