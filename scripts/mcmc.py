#!/usr/bin/env python3
"""Step 4: emcee on ESPEI 0.9.0's own log posterior (EmceeOptimizer.predict, ESPEI unmodified).

Env: ~/miniforge3/envs/calphad-uq (ESPEI 0.9.0, pycalphad 0.11.2, emcee 2.2.1).
Posterior: the same as the fit (crni.espei09_lnprob: ESPEI likelihood x fixed-centre Normal prior).
Initial walkers: theta_MAP + 0.1 * L z, z ~ N(0, I), with L L^T the Laplace covariance at the MAP
(fit/curvature_<set>.json); a draw is redrawn while its lnprob is not finite.
Checkpointing: chain.npy (steps, walkers, dim) and lnprob.npy every 10 steps; resumable. usage:
  mcmc.py <setname> <run_dir> <seed> <nwalkers> <nsteps> <nproc>
"""
import json
import os
import sys
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni

_F = None
_SET = None


def _init(setname):
    global _F, _SET
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)
    _SET = setname
    _F = crni.espei09_lnprob(setname)


def lnprob(theta):
    v = _F(theta)
    return v if np.isfinite(v) else -np.inf


def main():
    setname, run_dir, seed, nw, nsteps, nproc = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), \
        int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    import emcee  # 2.2.1, the version ESPEI 0.9.0 uses
    run_dir.mkdir(parents=True, exist_ok=True)
    cur = json.loads((crni.RUN / "fit" / f"curvature_{setname}.json").read_text())
    theta_map = np.asarray(cur["theta_map"])
    L = np.linalg.cholesky(np.asarray(cur["cov_theta"]))
    ndim = len(theta_map)
    chain_f, lp_f = run_dir / "chain.npy", run_dir / "lnprob.npy"
    rng = np.random.default_rng(seed)
    np.random.seed(seed)  # emcee 2 uses the global numpy RNG via its own RandomState
    with get_context("fork").Pool(nproc, initializer=_init, initargs=(setname,)) as pool:
        if chain_f.exists():
            chain, lps = list(np.load(chain_f)), list(np.load(lp_f))
            pos, lp0 = chain[-1], lps[-1]
            print(json.dumps({"resume_at": len(chain)}), flush=True)
        else:
            _init(setname)
            p0 = []
            while len(p0) < nw:
                cand = theta_map + 0.1 * L @ rng.standard_normal(ndim)
                if np.isfinite(lnprob(cand)):
                    p0.append(cand)
            pos, lp0 = np.array(p0), None
            np.save(run_dir / "p0.npy", pos)
            chain, lps = [], []
            meta = {"set": setname, "seed": seed, "nwalkers": nw, "ball": "theta_MAP + 0.1 * chol(Laplace cov) z",
                    "theta_map": theta_map.tolist(), "emcee": emcee.__version__,
                    "espei": __import__("espei").__version__, "pycalphad": __import__("pycalphad").__version__,
                    "move": "stretch a=2 (emcee 2 default)", "started": time.strftime("%Y-%m-%d %H:%M:%S")}
            (run_dir / "run.json").write_text(json.dumps(meta, indent=1))
        sampler = emcee.EnsembleSampler(nw, ndim, lnprob, pool=pool)
        sampler._random = np.random.RandomState(seed + len(chain))
        t0 = time.perf_counter()
        n0 = len(chain)
        acc = np.zeros(nw)
        for i, (p, lp, _) in enumerate(sampler.sample(pos, lnprob0=lp0, iterations=nsteps - n0, storechain=False)):
            if chain:
                acc += np.any(p != chain[-1], axis=1)
            chain.append(np.array(p)); lps.append(np.array(lp))
            it = len(chain)
            if it % 10 == 0 or it == nsteps:
                np.save(chain_f, np.array(chain)); np.save(lp_f, np.array(lps))
            if it % 25 == 0:
                print(json.dumps({"iter": it, "acc_since_start": round(float(np.mean(acc) / max(1, i)), 3),
                                  "lp_mean": float(np.mean(lp)), "lp_max": float(np.max(lp)),
                                  "s_per_iter": round((time.perf_counter() - t0) / (i + 1), 2)}), flush=True)
            if (run_dir / "STOP").exists():
                np.save(chain_f, np.array(chain)); np.save(lp_f, np.array(lps))
                print("STOP file found", flush=True)
                break


if __name__ == "__main__":
    main()
