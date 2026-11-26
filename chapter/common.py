"""Shared helpers for the chapter Section 7 figures (Cr-Ni set E, two-mode posterior).

Read-only use of ../scripts (crni, pdfast) and ../mcmc, ../results; everything written goes under chapter/.
pdfast.ref_grid caches next to the TDB in inputs/; to keep inputs untouched, ref_grid() here caches in chapter/cache.
Classification of a parameter vector (same as cr-ni-setH-20261106/scripts/mode_occupancy.py):
  ok    = pdfast.topology()["ok"]  (accepted Cr-Ni topology over 800-2200 K, 5 K steps)
  gap   = not ok and a same-phase (BCC_A2+BCC_A2) tie-line appears in the stable diagram
  other = not ok, no gap
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
CH = Path(__file__).resolve().parent
RUN = CH.parent
sys.path.insert(0, str(RUN / "scripts"))
import crni  # noqa: E402
import pdfast  # noqa: E402

DATA, FIG, CACHE = CH / "data", CH / "figures", CH / "cache"
for _d in (DATA, FIG, CACHE):
    _d.mkdir(exist_ok=True)
R = 8.314462618
TT = np.arange(800.0, 2201.0, 5.0)   # topology grid (as propagate.py / mode_crossings.py)
DIAG = json.loads((RUN / "results" / "setE_mcmc_diag.json").read_text())
BURN = {r["dir"]: r["burn"] for r in DIAG["runs"]}
RUNS = [r["dir"] for r in DIAG["runs"]]
X = pdfast.X
# decision: single-phase FCC_A1 at x_Ni = 0.52, 1473 K; decision map region near the bcc/fcc boundary
X_DEC, T_DEC = 0.52, 1473.0
MAP_X = np.round(np.arange(0.38, 0.6801, 0.01), 4)
MAP_T = np.arange(1200.0, 1601.0, 20.0)


def ref_grid(T):
    """G_ref+ideal (3, nT, nX); same as pdfast.ref_grid but cached in chapter/cache (or reuse pdfast's cache)."""
    T = np.asarray(T, float)
    tag = f"_ref_{len(T)}_{T[0]:.0f}_{T[-1]:.0f}_{pdfast.NX}.npy"
    old = Path(crni.TDB.with_suffix("").as_posix() + tag)
    if old.exists():
        return np.load(old)
    new = CACHE / ("cr-ni-model" + tag)
    if new.exists():
        return np.load(new)
    from pycalphad import Database, calculate
    db = Database(str(crni.TDB))
    pars = {f"VV{i:04d}": 0.0 for i in range(10)}
    pts = np.stack([1 - X, X], axis=1)
    out = np.empty((3, len(T), len(X)))
    for k, ph in enumerate(pdfast.PH):
        r = calculate(db, ["CR", "NI", "VA"], ph, T=T, P=101325, N=1, points=pts, parameters=pars)
        out[k] = np.asarray(r.GM.squeeze()).reshape(len(T), len(X))
    np.save(new, out)
    return out


def theta_of(name):
    f = {"accepted": "setE_orchid_sigmas_best.json", "gap": "setE_orchid_sigmas_gapmode.json"}[name]
    d = json.loads((RUN / "results" / f).read_text())
    return np.array([d["theta"][k] for k in crni.LABELS])


def classify(theta, ref=None):
    t = pdfast.topology(theta, TT, ref if ref is not None else ref_grid(TT))
    return ("ok" if t["ok"] else ("gap" if t["gaps"] else "other")), t


CLASSES = ["ok", "gap", "other"]


def bcc_params(theta, T):
    th = np.asarray(theta, float)
    return th[2] + th[3] * T, th[4] + th[5] * T


def bcc_spinodal(theta, T):
    """chemical spinodal of BCC_A2 alone: x_Ni where d2G/dx2 = 0 (RK, x = x_Ni, (xCr - xNi) = 1-2x)."""
    L0, L1 = bcc_params(theta, T)
    x = np.linspace(1e-4, 1 - 1e-4, 20001)
    # G_ex = x(1-x)[L0 + L1(1-2x)];  d2/dx2 = -2 L0 + L1(-6 + 12 x) ... computed numerically for safety
    gex = x * (1 - x) * (L0 + L1 * (1 - 2 * x))
    d2 = np.gradient(np.gradient(gex, x), x) + R * T / (x * (1 - x))
    neg = d2 < 0
    if not neg.any():
        return None
    i = np.where(neg)[0]
    return float(x[i[0]]), float(x[i[-1]])


def bcc_binodal(theta, T, ref1):
    """metastable BCC_A2 + BCC_A2 tie-line (lower hull of G_bcc alone); ref1: (3, nX) at T. None if no gap."""
    g = ref1[0] + pdfast.excess(theta, [T])[0, 0]
    h = pdfast._lower_hull(X, g)
    best = None
    for a, b in zip(h[:-1], h[1:]):
        if b - a > 3 and (best is None or X[b] - X[a] > best[1] - best[0]):
            best = (float(X[a]), float(X[b]))
    return best


def mu_cr_rel(theta, T, x, ref1):
    """RT ln a_Cr (reference: pure BCC_A2 Cr at T) at overall x_Ni = x from the stable lower hull at T.
    ref1: (3, nX) at T. Two-phase: tie-line intercept; single phase: tangent of that phase's G."""
    G = ref1 + pdfast.excess(theta, [T])[:, 0, :]
    idx = np.argmin(G, axis=0)
    gmin = G[idx, np.arange(G.shape[1])]
    h = pdfast._lower_hull(X, gmin)
    j = int(np.searchsorted(X, x))
    a = max(i for i in h if X[i] <= x)
    b = min(i for i in h if X[i] >= x)
    gcr0 = G[0, 0]  # BCC_A2 at x_Ni = 1e-6 (ideal term ~1e-4 J, negligible)
    if b - a > 1:  # tie-line
        slope = (gmin[b] - gmin[a]) / (X[b] - X[a])
        return float(gmin[a] - slope * X[a] - gcr0)
    k = idx[j]
    g = G[k]
    s = (g[j + 1] - g[j - 1]) / (X[j + 1] - X[j - 1])
    return float(g[j] - s * X[j] - gcr0)


def mu_cr_bcc_analytic(theta, T, x):
    L0, L1 = bcc_params(theta, T)
    xc = 1 - x
    return R * T * np.log(xc) + x ** 2 * (L0 + L1 * (3 * xc - x))


def region_name(segs, x):
    for kind, phs, xl, xr in segs:
        if xl - 1e-12 <= x <= xr + 1e-12:
            return "+".join(phs) if kind == 2 else phs[0]
    return "?"


def fcc_map(theta, refmap):
    """indicator of single-phase FCC_A1 on MAP_T x MAP_X; refmap = ref_grid(MAP_T)."""
    di = pdfast.diagram(theta, MAP_T, refmap)
    out = np.zeros((len(MAP_T), len(MAP_X)), bool)
    for i, segs in enumerate(di):
        for j, x in enumerate(MAP_X):
            out[i, j] = region_name(segs, x) == "FCC_A1"
    return out


def reconstruct_draws(ndraw):
    """the joint ensemble exactly as propagate.py builds it, plus (run, step, walker) of every draw."""
    post, meta = [], []
    n0 = DIAG["runs"][0]["steps"]
    for k, r in enumerate(DIAG["runs"]):
        ch = np.load(RUN / r["dir"] / "chain.npy")[:n0]
        p = ch[r["burn"]:]
        S, W = p.shape[:2]
        post.append(p.reshape(-1, p.shape[2]))
        st, wk = np.meshgrid(np.arange(r["burn"], r["burn"] + S), np.arange(W), indexing="ij")
        meta.append(np.stack([np.full(S * W, k), st.ravel(), wk.ravel()], 1))
    allp, allm = np.concatenate(post), np.concatenate(meta)
    idx = np.linspace(0, len(allp) - 1, ndraw).astype(int)
    return allp[idx], allm[idx]


def independent_of(joint, seed):
    rng = np.random.default_rng(seed)
    ind = joint.copy()
    for k in range(ind.shape[1]):
        ind[:, k] = ind[rng.permutation(len(ind)), k]
    return ind


def wilson(k, n, z=1.0):
    """Wilson score interval (z=1 gives a ~68 % interval) for k successes in n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cluster_se(ind, groups):
    """standard error of the mean of indicator ind with draws clustered by group (walker of a run)."""
    ind = np.asarray(ind, float)
    g = np.asarray(groups)
    u = np.unique(g)
    n = len(ind)
    m = ind.mean()
    s = sum((ind[g == v] - m).sum() ** 2 for v in u)
    return float(np.sqrt(s * len(u) / (len(u) - 1)) / n)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import style as st  # ORCHID-paper hero style (applies rcParams on import)
    return st
