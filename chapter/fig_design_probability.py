#!/usr/bin/env python3
"""fig_design_probability: P(single-phase fcc) at the decision state and across the decision map.
(a) P(fcc at x_Ni 0.52, 1473 K) and P(accepted topology) for the joint posterior, the independent marginals,
    the joint draws conditioned on each topology, and the Laplace approximation, with Monte Carlo errors
    (error bar = the larger of the binomial and the walker-cluster standard error; conditioned: Wilson z = 1).
(b-d) P(fcc) maps: joint, independent marginals, accepted-conditioned; (e) joint minus independent.
Reads data/chapter_numbers.json, data/fcc_maps.npz.
"""
import json

import numpy as np

import common as C
import pd_draw as D

st = C.style()
import matplotlib.pyplot as plt  # noqa: E402


def err(s):
    if s["n"] == 0:
        return np.nan, (0, 0)
    p = s["p"]
    if s["n"] < 200:
        lo, hi = s["wilson68"]
        return p, (p - lo, hi - p)
    e = max(s["se_binomial"], s.get("se_cluster_walker", 0))
    return p, (e, e)


def main():
    N = json.loads((C.DATA / "chapter_numbers.json").read_text())["decision_state"]
    M = np.load(C.DATA / "fcc_maps.npz")
    fig = plt.figure(figsize=(st.W2, 118 * st.MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    rows = [("Joint", N["joint"]["P_fcc"], N["joint"]["P_ok"], st.C_MODEL),
            ("Independent", N["indep"]["P_fcc"], N["indep"]["P_ok"], st.INK2),
            ("Joint | accepted", N["joint"]["P_fcc_given_ok"], None, st.C_MODEL),
            ("Joint | gap", N["joint"]["P_fcc_given_gap"], None, st.ACCENT),
            ("Joint | no Ni-rich bcc", N["joint"]["second_bcc_field_1200_1600K"]["P_fcc_given_no_second"], None,
             st.INK2),
            ("Laplace", N["laplace"]["P_fcc"], N["laplace"]["P_ok"], st.GREY_M)]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, pf, po, c) in zip(y, rows):
        p, e = err(pf)
        ax.errorbar(p, yi + 0.12, xerr=np.array(e)[:, None], fmt="o", ms=3, color=c, mfc=c, lw=0.7, capsize=1.5)
        ax.text(1.02, yi + 0.12, f"{p:.2f}", fontsize=6, va="center", color=c)
        if po is not None:
            q, f = err(po)
            ax.errorbar(q, yi - 0.18, xerr=np.array(f)[:, None], fmt="s", ms=2.8, color=c, mfc="white", lw=0.7,
                        capsize=1.5)
            ax.text(1.02, yi - 0.18, f"{q:.2f}", fontsize=6, va="center", color=c)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability")
    ax.plot([], [], "o", color=st.INK, ms=3, label="P(fcc) at decision state")
    ax.plot([], [], "s", color=st.INK, mfc="white", ms=2.8, label="P(accepted topology)")
    ax.legend(loc="lower left", fontsize=6, bbox_to_anchor=(0, 1.0), borderaxespad=0.1)
    st.panel_label(ax, "a")
    zpf = [(x, t) for x, t, ph, _ in D.zpf_points() if ph == "FCC_A1" and 1200 <= t <= 1600]
    maps = [("joint", "Joint posterior"), ("indep", "Independent marginals"), ("accepted", "Joint | accepted topology")]
    pos = [gs[0, 1], gs[0, 2], gs[1, 0]]
    for k, ((key, title), g) in enumerate(zip(maps, pos)):
        a = fig.add_subplot(g)
        im = a.pcolormesh(M["MAP_X"], M["MAP_T"], M[key], cmap="Oranges", vmin=0, vmax=1, shading="nearest",
                          rasterized=True)
        cs = a.contour(M["MAP_X"], M["MAP_T"], M[key], levels=[0.1, 0.5, 0.9], colors=st.INK, linewidths=0.5)
        a.clabel(cs, fontsize=5.5, fmt="%.1f")
        a.plot(*zip(*zpf), "o", ms=2, mfc="none", mec=st.C_EXP, mew=0.4)
        a.plot([C.X_DEC], [C.T_DEC], "+", color=st.INK, ms=5, mew=0.8)
        a.text(0.02, 0.98, title, transform=a.transAxes, fontsize=6.5, va="top",
               bbox=dict(fc="white", ec="none", pad=0.6))
        a.set_xlabel(r"$x_\mathrm{Ni}$")
        a.set_ylabel("Temperature (K)")
        st.panel_label(a, "bcd"[k])
    cb = fig.colorbar(im, ax=a, shrink=0.9, pad=0.01)
    cb.set_label("P(single-phase fcc)", fontsize=6.5)
    for k, (key, title) in enumerate((("indep", "Joint − independent"), ("accepted", "Joint − accepted"))):
        a = fig.add_subplot(gs[1, 1 + k])
        dmap = M["joint"] - M[key]
        im2 = a.pcolormesh(M["MAP_X"], M["MAP_T"], dmap, cmap="PuOr_r", vmin=-0.6, vmax=0.6, shading="nearest",
                           rasterized=True)
        a.contour(M["MAP_X"], M["MAP_T"], M["joint"], levels=[0.5], colors=st.INK, linewidths=0.5)
        a.plot([C.X_DEC], [C.T_DEC], "+", color=st.INK, ms=5, mew=0.8)
        a.text(0.02, 0.98, title, transform=a.transAxes, fontsize=6.5, va="top",
               bbox=dict(fc="white", ec="none", pad=0.6))
        a.set_xlabel(r"$x_\mathrm{Ni}$")
        a.set_ylabel("Temperature (K)")
        st.panel_label(a, "ef"[k])
    cb2 = fig.colorbar(im2, ax=a, shrink=0.9, pad=0.01)
    cb2.set_label("Difference in P(fcc)", fontsize=6.5)
    fig.savefig(C.FIG / "fig_design_probability.pdf")
    fig.savefig(C.FIG / "fig_design_probability.png", dpi=300)


if __name__ == "__main__":
    main()
