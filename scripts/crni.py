"""Shared definitions for the Cr-Ni rebuild (UQ CALPHAD chapter example).

Parameters theta (J/mol and J/mol/K), order VV0000..VV0009:
  L0 LIQ a,b | L0 BCC a,b | L1 BCC a,b | L0 FCC a,b | L1 FCC a,b
Optimiser coordinates q = theta / SCALE (a-terms /1000, b-terms /1), as in the paper.
Posterior = ESPEI likelihood (datasets of inputs/derived/<set>) x independent Normal prior
(centre = the paper's start values, sd 50 kJ/mol for a, 30 J/mol/K for b).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
INP = RUN / "inputs"
TDB = INP / "cr-ni-model.tdb"
NAMES = [f"VV{i:04d}" for i in range(10)]
LABELS = ["L0liq_a", "L0liq_b", "L0bcc_a", "L0bcc_b", "L1bcc_a", "L1bcc_b",
          "L0fcc_a", "L0fcc_b", "L1fcc_a", "L1fcc_b"]
PAPER_START = np.array([-13886.1, 4.632, 9545.6, -4.222, 47004.2, -11.221,
                        10224.7, -8.477, 36724.5, -15.324])
PAPER_OPT = np.array([-13552.9, 5.788, 14895.0, -6.353, 28769.5, -10.423,
                      4359.7, -5.957, 24477.4, -14.899])
SCALE = np.array([1000.0, 1.0] * 5)
PRIOR_CENTER = PAPER_START.copy()
PRIOR_SD = np.array([50000.0, 30.0] * 5)
PHASES = ["BCC_A2", "FCC_A1", "LIQUID"]
os.environ.setdefault("SETUPTOOLS_SCM_PRETEND_VERSION", "0.8.11.dev82")  # only used if the package lacks metadata


def datasets_dir(setname: str) -> Path:
    return INP / "derived" / setname / "datasets"


def phase_models() -> dict:
    return json.loads((INP / "CR-NI_phases.json").read_text())


def log_prior(theta):
    z = (np.asarray(theta) - PRIOR_CENTER) / PRIOR_SD
    return float(-0.5 * np.sum(z ** 2) - np.sum(np.log(PRIOR_SD * np.sqrt(2 * np.pi))))


def log_prior_grad(theta):
    return -(np.asarray(theta) - PRIOR_CENTER) / PRIOR_SD ** 2


# ---------------------------------------------------------------- ESPEI-grad (calphad-nuts env)
class GradTarget:
    """log posterior and gradient from ESPEI-grad residual objects, in q coordinates."""

    def __init__(self, setname: str, only=None):
        from espei.datasets import load_datasets, recursive_glob
        from espei.error_functions.context import setup_context
        from pycalphad import Database
        self.hinge = None
        side = INP / "derived" / setname / "hinge" / "CR-NI" / "cr-ni-model.json"
        if side.exists():  # PhaseForge+ hinge on excluded phases (set D); built before ESPEI touches the symbols
            from phaseforgeplus.utils.hinge import HingeOptions, HingePenalty
            sc = json.loads(side.read_text())
            keys = ("T", "x", "expected", "sigma", "tie_line", "confidence", "penalize")
            cells = [{k: c[k] for k in keys if k in c} for c in sc["cells"]]
            self.hinge = HingePenalty(Database(str(TDB)), NAMES, cells, sc["element"],
                                      HingeOptions(margin=float(sc["margin_J_per_mol_atom"])))
        self.convexity = None
        cv = INP / "derived" / setname / "convexity.json"
        if cv.exists():  # BCC convexity at observed single-phase BCC states (set F)
            from bcc_convexity import ConvexityPenalty
            c = json.loads(cv.read_text())
            self.convexity = ConvexityPenalty([(T, x) for T, x, _ in c["states"]], w=c["w_mole_fraction"],
                                              sigma=c["sigma_J_per_mol_atom"])
        dbf = Database(str(TDB))
        ds = load_datasets(sorted(recursive_glob(str(datasets_dir(setname)), "*.json")))
        ctx = setup_context(dbf, ds, symbols_to_fit=NAMES, phase_models=phase_models())
        assert list(ctx["symbols_to_fit"]) == NAMES, ctx["symbols_to_fit"]
        self.res = ctx["residual_objs"]
        if only:
            self.res = [r for r in self.res if type(r).__name__ in only]
        self.n = 0

    def parts(self, theta, gradient=True):
        out = {}
        for r in self.res:
            if gradient:
                v, g = r.get_likelihood_and_gradient(np.asarray(theta, float))
                out[type(r).__name__] = (float(v), np.asarray(g, float))
            else:
                out[type(r).__name__] = (float(r.get_likelihood(np.asarray(theta, float))), None)
        if self.hinge is not None:  # penalty is a negative log-likelihood
            p, gp = self.hinge(np.asarray(theta, float))
            out["HingePenalty"] = (-float(p), -np.asarray(gp, float))
        if self.convexity is not None:
            p, gp = self.convexity(np.asarray(theta, float))
            out["ConvexityPenalty"] = (-float(p), -np.asarray(gp, float))
        return out

    def __call__(self, q):
        theta = np.asarray(q, float) * SCALE
        self.n += 1
        parts = self.parts(theta)
        val = sum(v for v, _ in parts.values()) + log_prior(theta)
        g = sum(g for _, g in parts.values()) + log_prior_grad(theta)
        return val, g * SCALE  # d/dq = d/dtheta * SCALE


# ---------------------------------------------------------------- ESPEI 0.9.0 (calphad-uq env)
def espei09_lnprob(setname: str):
    """ESPEI 0.9.0's own EmceeOptimizer.predict with our fixed-centre Normal priors."""
    from espei.datasets import load_datasets, recursive_glob
    from espei.error_functions.context import setup_context
    from espei.optimizers.opt_mcmc import EmceeOptimizer
    from scipy.stats import norm
    from pycalphad import Database
    dbf = Database(str(TDB))
    ds = load_datasets(sorted(recursive_glob(str(datasets_dir(setname)), "*.json")))
    ctx = setup_context(dbf, ds, symbols_to_fit=NAMES, phase_models=phase_models())
    assert list(ctx["symbols_to_fit"]) == NAMES

    class _P:  # object with logpdf, as ESPEI's prior_rvs expects
        def __init__(self, loc, sd):
            self.rv = norm(loc=loc, scale=sd)

        def logpdf(self, x):
            return self.rv.logpdf(x)

    ctx["prior_rvs"] = [_P(c, s) for c, s in zip(PRIOR_CENTER, PRIOR_SD)]

    def lnprob(theta):
        return float(EmceeOptimizer.predict(np.asarray(theta, dtype=np.float64), **ctx))

    def parts(theta):
        return {type(r).__name__: float(r.get_likelihood(np.asarray(theta, float))) for r in ctx["residual_objs"]}

    lnprob.parts = parts
    lnprob.ctx = ctx
    return lnprob
