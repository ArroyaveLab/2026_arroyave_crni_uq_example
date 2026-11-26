"""Check: pdfast (convex-hull phase diagram) against pycalphad equilibrium at the published optimum."""
import sys, time, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); import numpy as np, crni, pdfast
from pycalphad import Database, equilibrium, variables as v
T = np.arange(800, 2201, 5.0)
ref = pdfast.ref_grid(T, crni.TDB)
th = crni.PAPER_OPT
t0=time.perf_counter(); top = pdfast.topology(th, T, ref); dt=time.perf_counter()-t0
print("topology", json.dumps(top)[:600], "s", round(dt,2))
db = Database(str(crni.TDB)); pars = dict(zip(crni.NAMES, th))
for t, x in [(1000, 0.40), (1000, 0.62), (1473, 0.40), (1473, 0.52), (1473, 0.70), (1650, 0.5), (1650, 0.3), (1900, 0.3)]:
    eq = equilibrium(db, ["CR","NI","VA"], crni.PHASES, {v.T: t, v.P: 101325, v.X("NI"): x, v.N: 1}, parameters=pars)
    ph = sorted(set(p for p in np.asarray(eq.Phase).ravel() if p))
    print(t, x, "pycalphad:", "+".join(ph), "| hull:", pdfast.phase_at(th, t, x, pdfast.ref_grid([t], crni.TDB)))
b = pdfast.boundaries(th, np.array([1000., 1200., 1473.]), pdfast.ref_grid(np.array([1000., 1200., 1473.]), crni.TDB))
print({k: np.round(val, 4).tolist() for k, val in b.items() if k in ("bcc_bf", "fcc_bf")})
for t in (1000., 1200., 1473.):
    eq = equilibrium(db, ["CR","NI","VA"], crni.PHASES, {v.T: t, v.P: 101325, v.X("NI"): 0.4, v.N: 1}, parameters=pars)
    print(t, np.asarray(eq.Phase).ravel()[:2], np.round(np.asarray(eq.X.sel(component="NI")).ravel()[:2], 4))
