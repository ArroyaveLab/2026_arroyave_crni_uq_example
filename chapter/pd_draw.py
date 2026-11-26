"""Computed Cr-Ni phase-diagram drawing for the chapter figures (pdfast hull construction, 1 K steps).

Every line is read from the computed lower convex hull: two-phase fields are hull tie-lines, invariants are drawn at
the temperatures where the set of two-phase fields changes, from the computed end compositions. Nothing is drawn
by hand. Also: the metastable BCC_A2-only binodal and chemical spinodal, and the ZPF data of set E.
"""
import json

import numpy as np

import common as C

P = C.pdfast
NMIN = 2.5 / (P.NX - 1)  # pdfast's rule: a grid-cell step at a G crossing is not a two-phase field
PAIRS = ["BCC_A2+FCC_A1", "BCC_A2+LIQUID", "FCC_A1+LIQUID", "BCC_A2+BCC_A2"]
PH_MARK = {"BCC_A2": "s", "FCC_A1": "o", "LIQUID": "^"}


def computed(theta, T):
    """per-T two-phase segments; returns dict pair -> (xl, xr) arrays (NaN if absent), and the T grid."""
    ref = C.ref_grid(T)
    di = P.diagram(theta, T, ref)
    out = {p: (np.full(len(T), np.nan), np.full(len(T), np.nan)) for p in PAIRS}
    for i, segs in enumerate(di):
        for kind, phs, xl, xr in segs:
            if kind != 2 or (xr - xl) < NMIN:
                continue
            key = "+".join(sorted(phs)) if phs[0] != phs[1] else phs[0] + "+" + phs[1]
            if key in out:
                if key == "BCC_A2+BCC_A2" and np.isfinite(out[key][0][i]):
                    continue
                out[key][0][i], out[key][1][i] = xl, xr
    return out


def invariants(seg, T):
    """horizontal invariant lines from the computed segments: (T, x_left, x_right, label)."""
    inv = []
    bf, bl, fl, bb = (seg[p] for p in PAIRS)
    liq = np.isfinite(bl[0]) | np.isfinite(fl[0])
    if liq.any():  # eutectic: lowest T with liquid; L -> BCC + FCC
        i = int(np.argmax(liq))
        j = max(i - 1, 0)
        inv.append((T[i], bf[0][j], bf[1][j], "eutectic"))
    gp = np.isfinite(bb[0])
    if gp.any():  # lowest T of the stable BCC+BCC gap: BCC2 -> BCC1 + FCC
        i = int(np.argmax(gp))
        inv.append((T[i], bb[0][i], bf[1][i], "gap invariant"))
    return inv


def metastable_bcc(theta, T):
    ref = C.ref_grid(T)
    bn = np.full((len(T), 2), np.nan)
    sp = np.full((len(T), 2), np.nan)
    for i, t in enumerate(T):
        b = C.bcc_binodal(theta, t, ref[:, i, :])
        if b:
            bn[i] = b
        s = C.bcc_spinodal(theta, t)
        if s:
            sp[i] = s
    return bn, sp


def gap_excursion(theta, T):
    """stable-gap temperature range, metastable binodal top, and the signed depth (J/mol-atom) of the Ni-rich
    binodal end below the hull of FCC, LIQUID and the Cr-rich BCC branch (positive = the gap is stable)."""
    ref = C.ref_grid(T)
    G = ref + P.excess(theta, T)
    seg = computed(theta, T)
    st = T[np.isfinite(seg["BCC_A2+BCC_A2"][0])]
    tops, depth = [], []
    for i, t in enumerate(T):
        bn = C.bcc_binodal(theta, t, ref[:, i, :])
        if not bn:
            continue
        tops.append(t)
        x1, x2 = bn
        j2 = int(np.argmin(abs(P.X - x2)))
        g = np.minimum(G[1, i], G[2, i]).copy()
        jj = P.X <= x1 + 1e-12
        g[jj] = np.minimum(g[jj], G[0, i][jj])
        h = P._lower_hull(P.X, g)
        a = max(k for k in h if k <= j2)
        b = min(k for k in h if k >= j2)
        hn = g[a] if a == b else g[a] + (g[b] - g[a]) * (P.X[j2] - P.X[a]) / (P.X[b] - P.X[a])
        depth.append((float(t), float(hn - G[0, i, j2])))
    dmax = max(depth, key=lambda z: z[1]) if depth else None
    return {"stable_gap_T": [float(st.min()), float(st.max())] if len(st) else None,
            "binodal_top_K": float(max(tops)) if tops else None,
            "max_signed_depth_J": dmax, "depth_by_T": depth}


def zpf_points():
    pts = []
    for f in sorted(C.crni.datasets_dir("setE_orchid_sigmas").glob("zpf_CR*.json")):
        d = json.loads(f.read_text())
        Ts = d["conditions"]["T"]
        Ts = Ts if isinstance(Ts, list) else [Ts] * len(d["values"])
        for t, reg in zip(Ts, d["values"]):
            for ph, comp, (x,) in reg:
                if x is None:
                    continue
                pts.append((x if comp == ["NI"] else 1 - x, t, ph, d["reference"]))
    return pts


def draw(ax, theta, T, color, lw=0.8, meta=True, data=True, st=None, mcolor=None):
    seg = computed(theta, T)
    for p in PAIRS:
        for a in seg[p]:
            ax.plot(a, T, "-", color=color, lw=lw, solid_capstyle="butt")
    for t, xl, xr, _ in invariants(seg, T):
        ax.plot([xl, xr], [t, t], "-", color=color, lw=lw)
    if meta:
        bn, sp = metastable_bcc(theta, T)
        mc = mcolor or color
        for k in (0, 1):
            ax.plot(bn[:, k], T, "--", color=mc, lw=0.7, dashes=(3, 1.5))
            ax.plot(sp[:, k], T, ":", color=mc, lw=0.8)
        # close the domes at their tops
        ok = np.isfinite(bn[:, 0])
        if ok.any():
            i = np.where(ok)[0][-1]
            ax.plot(bn[i], [T[i]] * 2, "--", color=mc, lw=0.7, dashes=(3, 1.5))
    if data:
        for x, t, ph, ref in zpf_points():
            ax.plot(x, t, PH_MARK.get(ph, "o"), ms=2.4, mfc="none", mec=st.C_EXP, mew=0.5, zorder=3)
    return seg
