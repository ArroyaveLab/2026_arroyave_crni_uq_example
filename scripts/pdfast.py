"""Fast exact-on-grid Cr-Ni phase diagram from lower convex hulls (no equilibrium solver).

G_phase(x,T) = G_ref+ideal (computed once with pycalphad `calculate`, all VV = 0)
             + x_Cr x_Ni [L0 + L1 (x_Cr - x_Ni)]   (pycalphad Redlich-Kister convention)
For each T the lower convex hull of the three phases' G(x) samples is built; hull edges
joining non-adjacent grid points are two-phase tie-lines (same phase twice = miscibility gap).
"""
from __future__ import annotations

import numpy as np
from pathlib import Path

PH = ["BCC_A2", "FCC_A1", "LIQUID"]
NX = 1201
X = np.concatenate([[1e-6], np.linspace(0, 1, NX)[1:-1], [1 - 1e-6]])  # x_Ni


def ref_grid(T, tdb):
    """G_ref+ideal for each phase on (T, X); cached on disk."""
    T = np.asarray(T, float)
    cache = Path(tdb).with_suffix("").as_posix() + f"_ref_{len(T)}_{T[0]:.0f}_{T[-1]:.0f}_{NX}.npy"
    if Path(cache).exists():
        return np.load(cache)
    from pycalphad import Database, calculate
    db = Database(str(tdb))
    pars = {f"VV{i:04d}": 0.0 for i in range(10)}
    pts = np.stack([1 - X, X], axis=1)  # site fractions (CR, NI)
    out = np.empty((3, len(T), len(X)))
    for k, ph in enumerate(PH):
        r = calculate(db, ["CR", "NI", "VA"], ph, T=T, P=101325, N=1, points=pts, parameters=pars)
        out[k] = np.asarray(r.GM.squeeze()).reshape(len(T), len(X))
    np.save(cache, out)
    return out


def excess(theta, T):
    """(3, nT, nX) excess Gibbs energy for BCC, FCC, LIQ."""
    th = np.asarray(theta, float)
    T = np.asarray(T, float)[:, None]
    xc, xn = 1 - X[None, :], X[None, :]
    def rk(L0, L1):
        return xc * xn * (L0 + L1 * (xc - xn))
    liq = rk(th[0] + th[1] * T, 0.0)
    bcc = rk(th[2] + th[3] * T, th[4] + th[5] * T)
    fcc = rk(th[6] + th[7] * T, th[8] + th[9] * T)
    return np.stack([bcc, fcc, liq])


def _lower_hull(x, g):
    """indices of lower convex hull of points sorted by x (monotone chain)."""
    h = []
    for i in range(len(x)):
        while len(h) >= 2:
            a, b = h[-2], h[-1]
            if (x[b] - x[a]) * (g[i] - g[a]) - (g[b] - g[a]) * (x[i] - x[a]) <= 0:
                h.pop()
            else:
                break
        h.append(i)
    return h


def regions_at(G):
    """G: (3, nX) at one T. Returns list of segments [(kind, phases, x_left, x_right)]
    kind 1 = single phase, 2 = two-phase tie-line. Ordered in x_Ni."""
    # take the minimum phase at each x first (hull of the lower envelope suffices)
    idx = np.argmin(G, axis=0)
    gmin = G[idx, np.arange(G.shape[1])]
    h = _lower_hull(X, gmin)
    segs = []
    for a, b in zip(h[:-1], h[1:]):
        pa, pb = PH[idx[a]], PH[idx[b]]
        if b - a == 1 and pa == pb:
            if segs and segs[-1][0] == 1 and segs[-1][1] == (pa,):
                segs[-1] = (1, (pa,), segs[-1][2], X[b])
            else:
                segs.append((1, (pa,), X[a], X[b]))
        elif b - a <= 2 and pa != pb and False:
            pass
        else:
            segs.append((2, (pa, pb), X[a], X[b]))
    # merge spurious tiny two-phase steps (one grid cell between different phases at a crossing)
    return segs


def diagram(theta, T, ref):
    G = ref + excess(theta, T)
    return [regions_at(G[:, i, :]) for i in range(len(T))]


def seq_string(segs, min_width=0.0):
    names = []
    for kind, phs, xl, xr in segs:
        if kind == 2 and (xr - xl) < 2.5 / (NX - 1) and phs[0] != phs[1]:
            continue  # a grid-cell step at a G crossing is not a two-phase field
        s = "+".join(phs) if kind == 2 else phs[0]
        if not names or names[-1] != s:
            names.append(s)
    return names


def topology(theta, T, ref, t_eut_expected=(1560.0, 1680.0)):
    """Check the accepted Cr-Ni topology over T. Returns dict."""
    di = diagram(theta, T, ref)
    bad, gaps, seqs = [], [], []
    has_liq_low = None
    for t, segs in zip(T, di):
        s = seq_string(segs)
        seqs.append(s)
        for kind, phs, xl, xr in segs:
            if kind == 2 and phs[0] == phs[1] and (xr - xl) > 2.5 / (NX - 1):
                gaps.append((float(t), phs[0], float(xl), float(xr)))
    # eutectic estimate: lowest T at which LIQUID appears anywhere
    liq_T = [float(t) for t, s in zip(T, seqs) if any("LIQUID" in z for z in s)]
    tliq_min = min(liq_T) if liq_T else np.inf
    bf_T = [float(t) for t, s in zip(T, seqs) if "BCC_A2+FCC_A1" in s or "FCC_A1+BCC_A2" in s]
    tbf_max = max(bf_T) if bf_T else -np.inf
    ok = True
    for t, s in zip(T, seqs):
        if t < tliq_min:
            want = [["BCC_A2", "BCC_A2+FCC_A1", "FCC_A1"]]
        elif t < 1728.0 - 1:
            want = [["BCC_A2", "BCC_A2+LIQUID", "LIQUID", "LIQUID+FCC_A1", "FCC_A1"],
                    ["BCC_A2", "BCC_A2+LIQUID", "LIQUID", "FCC_A1+LIQUID", "FCC_A1"]]
        elif t < 2180.0 - 1:
            want = [["BCC_A2", "BCC_A2+LIQUID", "LIQUID"]]
        else:
            want = [["LIQUID"]]
        if s not in want and not (abs(t - tliq_min) < 30 or abs(t - 1728) < 10 or abs(t - 2180) < 3):
            ok = False
            bad.append((float(t), s))
    ok = ok and not gaps and (t_eut_expected[0] <= tliq_min <= t_eut_expected[1]) and tbf_max <= tliq_min + 30
    return {"ok": bool(ok), "T_liquid_min": tliq_min, "T_bcc_fcc_max": tbf_max,
            "gaps": gaps[:20], "n_bad": len(bad), "bad": bad[:20]}


def boundaries(theta, T, ref):
    """BCC/(BCC+FCC) and (BCC+FCC)/FCC compositions (x_Ni) per T; liquidus/solidus edges."""
    di = diagram(theta, T, ref)
    out = {"bcc_bf": [], "fcc_bf": [], "liq_bccside": [], "liq_fccside": [],
           "sol_bcc": [], "sol_fcc": []}
    for segs in di:
        r = dict((k, np.nan) for k in out)
        for kind, phs, xl, xr in segs:
            if kind != 2:
                continue
            if set(phs) == {"BCC_A2", "FCC_A1"}:
                r["bcc_bf"], r["fcc_bf"] = xl, xr
            elif set(phs) == {"BCC_A2", "LIQUID"}:
                r["sol_bcc"], r["liq_bccside"] = xl, xr
            elif set(phs) == {"FCC_A1", "LIQUID"}:
                r["liq_fccside"], r["sol_fcc"] = xl, xr
        for k in out:
            out[k].append(r[k])
    return {k: np.asarray(v) for k, v in out.items()}


def phase_at(theta, t, x, ref1):
    """name of the region containing x_Ni = x at a single T (ref1: (3,1,nX))."""
    segs = regions_at((ref1 + excess(theta, [t]))[:, 0, :])
    for kind, phs, xl, xr in segs:
        if xl - 1e-12 <= x <= xr + 1e-12:
            return "+".join(phs) if kind == 2 else phs[0]
    return "?"
