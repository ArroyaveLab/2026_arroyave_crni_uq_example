#!/usr/bin/env python3
"""Step 2: multistart MAP fit with ESPEI-grad analytic gradients (L-BFGS-B in q coordinates).

Env: ~/.venvs/espei-pr268 (ESPEI-grad origin/Gradient-based_update 9039ea2, pycalphad 0.11.3.dev35).
Fallback when the analytic path returns a non-finite value or gradient: value-only ESPEI-grad
path (`get_likelihood`) and a central finite-difference gradient; every such event is logged.
usage: fit.py <setname> <nproc> [seed]
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

_T = {}


def target(setname):
    if setname not in _T:
        _T[setname] = crni.GradTarget(setname)
    return _T[setname]


def value_only(t, theta):
    h = -float(t.hinge(np.asarray(theta, float))[0]) if t.hinge is not None else 0.0
    if getattr(t, "convexity", None) is not None:
        h -= float(t.convexity(np.asarray(theta, float))[0])
    return sum(float(r.get_likelihood(theta)) for r in t.res) + crni.log_prior(theta) + h


def run(job):
    warnings.filterwarnings("ignore")
    from scipy.optimize import minimize
    setname, name, q0 = job
    t = target(setname)
    events = []
    nev = [0]

    def f(q):
        nev[0] += 1
        theta = q * crni.SCALE
        try:
            val, g = t(q)
            val = float(val)
        except Exception as e:  # noqa
            val, g = -np.inf, None
            events.append({"q": q.tolist(), "error": repr(e)[:200]})
        if np.isfinite(val) and g is not None and np.all(np.isfinite(g)):
            return -val, -g
        # isolated equilibrium failures (seen: one activity point -> nan in BOTH ESPEI 0.9.0 and
        # ESPEI-grad): retry at a tiny jitter before any other fallback
        rngj = np.random.default_rng(nev[0])
        for rel in (1e-6, 1e-5, 1e-4, 1e-3):
            qj = q * (1 + rel * rngj.standard_normal(len(q)))
            try:
                vj, gj = t(qj)
                vj = float(vj)
            except Exception:  # noqa
                continue
            if np.isfinite(vj) and np.all(np.isfinite(gj)):
                events.append({"q": q.tolist(), "analytic_value": val, "jitter_ok": rel})
                return -vj, -gj
        # fallback
        v2 = value_only(t, theta)
        ev = {"q": q.tolist(), "analytic_value": val, "value_only": v2}
        if not np.isfinite(v2):
            events.append(ev)
            return 1e12, np.zeros_like(q)
        g2 = np.zeros_like(q)
        for i in range(len(q)):
            h = 1e-3 * max(1.0, abs(q[i]))
            qp, qm = q.copy(), q.copy()
            qp[i] += h; qm[i] -= h
            g2[i] = (value_only(t, qp * crni.SCALE) - value_only(t, qm * crni.SCALE)) / (2 * h)
        ev["fd_gradient_used"] = True
        events.append(ev)
        return -v2, -g2

    t0 = time.perf_counter()
    q = np.asarray(q0, float)
    msgs = []
    for _ in range(4):
        r = minimize(f, q, jac=True, method="L-BFGS-B",
                     options={"maxiter": 400, "maxfun": 800, "ftol": 1e-12, "gtol": 1e-6, "maxcor": 20})
        moved = float(np.linalg.norm(r.x - q))
        q = r.x
        msgs.append(str(r.message))
        if moved < 1e-5:
            break
    val, g = t(q)
    return {"set": setname, "start": name, "q0": list(map(float, q0)), "q": q.tolist(),
            "theta": (q * crni.SCALE).tolist(), "logpost": float(val),
            "grad_norm_q": float(np.linalg.norm(g)), "nfev": nev[0], "messages": msgs,
            "nonfinite_events": events, "seconds": time.perf_counter() - t0}


def starts(seed):
    rng = np.random.default_rng(seed)
    qs = crni.PAPER_START / crni.SCALE
    qo = crni.PAPER_OPT / crni.SCALE
    jobs = [("paper_opt", qo), ("paper_start", qs)]
    sd_q = np.array([10.0, 5.0] * 5)  # 10 kJ/mol on a, 5 J/mol/K on b
    for i in range(3):
        jobs.append((f"near_opt_{i}", qo + sd_q * rng.standard_normal(10)))
    for i in range(3):
        jobs.append((f"near_start_{i}", qs + sd_q * rng.standard_normal(10)))
    return jobs


def main():
    setname, nproc = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 20261105
    out = crni.RUN / "fit" / f"multistart_{setname}.json"
    if os.environ.get("FIT_POLISH_FROM"):  # polish the endpoints of an earlier multistart file
        prev = json.loads(Path(os.environ["FIT_POLISH_FROM"]).read_text())
        jobs = [(setname, "polish_" + r["start"], np.asarray(r["q"])) for r in prev]
        out = out.with_name(f"polish_{setname}.json")
    else:
        jobs = [(setname, n, q) for n, q in starts(seed)]
    if os.environ.get("FIT_EXTRA_STARTS"):  # extra named starts: JSON {name: theta}
        extra = json.loads(Path(os.environ["FIT_EXTRA_STARTS"]).read_text())
        jobs = [(setname, n, np.asarray(t) / crni.SCALE) for n, t in extra.items()] + \
            ([] if os.environ.get("FIT_ONLY_EXTRA") else jobs[:max(0, 8 - len(extra))])
    res = []
    with get_context("spawn").Pool(nproc) as pool:
        for r in pool.imap_unordered(run, jobs):
            res.append(r)
            print(json.dumps({k: r[k] for k in ("set", "start", "logpost", "grad_norm_q", "nfev", "seconds")}
                             | {"n_nonfinite": len(r["nonfinite_events"])}), flush=True)
            res.sort(key=lambda z: -z["logpost"])
            out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
