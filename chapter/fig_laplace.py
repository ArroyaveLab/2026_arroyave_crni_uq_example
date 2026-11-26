#!/usr/bin/env python3
"""fig_laplace: the curvature-based (Laplace) approximation at the accepted optimum against the MCMC posterior.
(a) 90 % bands (5-95 %) of the BCC/(BCC+FCC) and (BCC+FCC)/FCC boundaries: Laplace, MCMC joint, and MCMC
    conditioned on the accepted topology, with the accepted optimum and the ZPF data.
(b) marginal standard deviations: MCMC (pooled post-burn-in, results/setE_mcmc_diag.json) over Laplace.
Laplace covariance: cov_theta of cr-ni-setH-20261106/inputs/setE_spark/curvature_setE_run1_b557.json (curvature.py:
central differences of the ESPEI-grad analytic gradient at the accepted optimum, steps 1e-3 and 3e-3; the raw
finite-difference -H has one negative eigenvalue, -0.005, so its 3 softest eigenvalues are replaced by line-scan
curvatures, soft_scan.py / repair_cov.py). Writes data/laplace_summary.json.
"""
import json

import numpy as np

import common as C
import pd_draw as D

st = C.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

CURV = C.RUN / "inputs/setE_spark/curvature_setE_run1_b557.json"
NICE = [r"$^{0}L^\mathrm{liq}$ a", r"$^{0}L^\mathrm{liq}$ b", r"$^{0}L^\mathrm{bcc}$ a", r"$^{0}L^\mathrm{bcc}$ b",
        r"$^{1}L^\mathrm{bcc}$ a", r"$^{1}L^\mathrm{bcc}$ b", r"$^{0}L^\mathrm{fcc}$ a", r"$^{0}L^\mathrm{fcc}$ b",
        r"$^{1}L^\mathrm{fcc}$ a", r"$^{1}L^\mathrm{fcc}$ b"]


def main():
    B = np.load(C.DATA / "boundary_bands.npz")
    T = B["TT"]
    cur = json.loads(CURV.read_text())
    fig = plt.figure(figsize=(st.W2, 88 * st.MM), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1])
    ax = fig.add_subplot(gs[0])
    sty = [("laplace", st.GREY_M, 0.55, "Laplace at the accepted optimum"),
           ("joint", st.ACCENT, 0.25, "MCMC, joint (both modes)"),
           ("accepted", st.C_MODEL, 0.35, "MCMC, accepted topology only")]
    for name, c, al, lab in sty:
        for b in ("bcc_bf", "fcc_bf"):
            q = B[f"{name}_{b}"]
            m = (T >= 1200) & (T <= 1615)
            if name == "joint":  # very wide and bimodal: 5 % and 95 % lines only
                for k in (0, 2):
                    ax.plot(q[k][m], T[m], color=c, lw=0.8, ls=(0, (3, 1.5)))
            else:
                ax.fill_betweenx(T[m], q[0][m], q[2][m], color=c, alpha=al, lw=0)
    th = C.theta_of("accepted")
    bd = C.pdfast.boundaries(th, T, C.ref_grid(T))
    m = (T >= 1200) & (T <= 1615)
    for b in ("bcc_bf", "fcc_bf"):
        ax.plot(bd[b][m], T[m], color=st.INK, lw=0.6)
    for x, t, ph, _ in D.zpf_points():
        if 1200 <= t <= 1650 and ph != "LIQUID":
            ax.plot(x, t, D.PH_MARK[ph], ms=2.2, mfc="none", mec=st.C_EXP, mew=0.45)
    ax.plot([C.X_DEC], [C.T_DEC], "+", color=st.INK, ms=5, mew=0.8)
    ax.set_xlim(0, 0.7)
    ax.set_ylim(1200, 1650)
    ax.set_xlabel(r"$x_\mathrm{Ni}$")
    ax.set_ylabel("Temperature (K)")
    hs = [Patch(color=sty[0][1], alpha=0.75, label=sty[0][3] + " (5–95 %)"),
          Line2D([], [], color=sty[1][1], lw=0.8, ls=(0, (3, 1.5)), label=sty[1][3] + " (5 % and 95 %)"),
          Patch(color=sty[2][1], alpha=0.55, label=sty[2][3] + " (5–95 %)"),
          Line2D([], [], color=st.INK, lw=0.6, label="Accepted optimum")]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=6)
    st.panel_label(ax, "a")
    diag = C.DIAG
    sd_mc = np.array(diag["post_sd"])
    sd_l = np.array(cur["sd_theta"])
    a2 = fig.add_subplot(gs[1])
    y = np.arange(10)[::-1]
    a2.barh(y, sd_mc / sd_l, color=st.C_MODEL, height=0.6)
    a2.axvline(1, color=st.INK, lw=0.5)
    a2.set_xlim(0, 2)
    a2.set_yticks(y)
    a2.set_yticklabels(NICE)
    a2.set_xlabel("Posterior sd: MCMC / Laplace")
    st.panel_label(a2, "b")
    fig.savefig(C.FIG / "fig_laplace.pdf")
    fig.savefig(C.FIG / "fig_laplace.png", dpi=300)
    i = int(np.argmin(abs(T - 1473)))
    out = {"method": "Laplace: N(theta_opt, cov_theta) from " + str(CURV.relative_to(C.RUN)) + "; " + cur.get("init_cov_note", ""),
           "eig_negH_q_fd": cur["eig_negH_q_fd"], "eig_negH_q_used": cur["eig_negH_q_used"],
           "rel_stepsize_diff": cur["rel_stepsize_diff"], "rel_asymmetry": cur["rel_asymmetry"],
           "sd_laplace": sd_l.tolist(), "sd_mcmc": sd_mc.tolist(), "ratio_mcmc_over_laplace": (sd_mc / sd_l).tolist(),
           "width90_1473K": {f"{n}_{b}": float(B[f"{n}_{b}"][2][i] - B[f"{n}_{b}"][0][i])
                             for n in ("laplace", "joint", "accepted", "indep") for b in ("bcc_bf", "fcc_bf")}}
    (C.DATA / "laplace_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("ratio_mcmc_over_laplace", "width90_1473K")}, indent=1))


if __name__ == "__main__":
    main()
