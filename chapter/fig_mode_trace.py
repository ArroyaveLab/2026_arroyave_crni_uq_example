#!/usr/bin/env python3
"""fig_mode_trace: decision-relevant convergence evidence for the two set E runs.
(a, b) topology class (accepted / gap / other) of every walker every K steps, burn-in marked;
(c) fraction of walkers in each class along each run; (d) P(single-phase fcc) map difference run 1 - run 2
(from data/ensembles.npz, joint draws split by run), with the decision state.
Per-run occupancy: mean of the post-burn-in indicator; error = sqrt(p(1-p)/ESS), ESS from the walker-averaged
autocorrelation of the indicator (initial positive sequence, as mode_crossings.py). Writes data/mode_trace_summary.json.
usage: fig_mode_trace.py [K]
"""
import json
import sys

import numpy as np

import common as C

st = C.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

COL = [st.C_MODEL, st.ACCENT, st.GREY_M]  # ok, gap, other
LAB = ["Accepted topology", "bcc + bcc gap", "Other"]


def ind_stats(ind):
    """ind: (walkers, n) 0/1. mean, tau (in thinned steps), ESS, se."""
    p = ind.mean()
    x = ind - p
    n = ind.shape[1]
    acf = np.array([np.mean([np.dot(xw[:n - k], xw[k:]) / n for xw in x]) for k in range(n // 2)])
    acf = acf / acf[0] if acf[0] > 0 else acf
    cut = int(np.argmax(acf < 0)) if np.any(acf < 0) else len(acf)
    tau = max(1.0, 1 + 2 * np.sum(acf[1:cut]))
    ess = ind.size / tau
    return float(p), float(tau), float(ess), float(np.sqrt(p * (1 - p) / ess))


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    z = np.load(C.DATA / f"mode_trace_K{K}.npz")
    summ = {"K": K, "classification": C.__doc__.split("Classification")[1].strip(), "runs": {}}
    fig = plt.figure(figsize=(st.W2, 112 * st.MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    cmap = ListedColormap(COL)
    axc = fig.add_subplot(gs[0, 1])
    for r in (1, 2):
        cls, steps, burn = z[f"run{r}_cls"], z[f"run{r}_steps"], int(z[f"run{r}_burn"])
        ax = fig.add_subplot(gs[r - 1, 0])
        ax.imshow(cls, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest",
                  extent=[steps[0] - K / 2, steps[-1] + K / 2, cls.shape[0] + 0.5, 0.5])
        ax.axvline(burn, color=st.INK, lw=0.8, ls="--")
        ax.text(burn - 80, 19.6, "end of burn-in", fontsize=6, va="bottom", ha="right", bbox=dict(fc="white", ec="none", pad=0.4))
        ax.set_ylabel(f"Run {r}: walker")
        ax.set_yticks([1, 10, 20])
        ax.set_xlabel("Step")
        post = steps >= burn
        ok = (cls[:, post] == 0).astype(float)
        gp = (cls[:, post] == 1).astype(float)
        p, tau, ess, se = ind_stats(ok)
        pg, taug, essg, seg = ind_stats(gp)
        cross = int(np.abs(np.diff(ok, axis=1)).sum())
        summ["runs"][f"run{r}"] = {"burn": burn, "n_thinned_steps_post_burn": int(post.sum()),
                                   "P_ok": p, "P_ok_se": se, "tau_ok_in_K_steps": tau, "ess_ok": ess,
                                   "P_gap": pg, "P_gap_se": seg, "P_other": float((cls[:, post] == 2).mean()),
                                   "crossings_ok_vs_not": cross,
                                   "walkers_never_ok_post_burn": int((ok.sum(1) == 0).sum()),
                                   "P_ok_first_1000_steps": float((cls[:, steps < 1000] == 0).mean())}
        frac = np.stack([(cls == k).mean(0) for k in range(3)])
        for k in range(3):
            axc.plot(steps, frac[k], color=COL[k], lw=0.7, ls="-" if r == 1 else (0, (2.5, 1.2)))
        ax.text(1.0, 1.01,
                f"post-burn-in: P(accepted) {p:.3f} ± {se:.3f}, P(gap) {pg:.3f} ± {seg:.3f}; "
                f"{cross} crossings; indicator ESS {ess:.0f}",
                fontsize=6, ha="right", va="bottom", color=st.INK, transform=ax.transAxes)
        st.panel_label(ax, "ab"[r - 1])
    axc.set_xlabel("Step")
    axc.set_ylabel("Fraction of walkers")
    axc.set_ylim(0, 1)
    axc.plot([], [], color=st.INK2, lw=0.7, label="Run 1")
    axc.plot([], [], color=st.INK2, lw=0.7, ls=(0, (2.5, 1.2)), label="Run 2")
    axc.legend(loc="center right", fontsize=6)
    st.panel_label(axc, "c")
    # (d) per-run P(fcc) maps from the joint draws
    e = np.load(C.DATA / "ensembles.npz")
    run = e["joint_meta"][:, 0]
    fm = e["joint_fccmap"].astype(float)
    p1, p2 = fm[run == 0].mean(0), fm[run == 1].mean(0)
    n1, n2 = (run == 0).sum(), (run == 1).sum()
    axd = fig.add_subplot(gs[1, 1])
    lim = 0.3
    im = axd.pcolormesh(C.MAP_X, C.MAP_T, p1 - p2, cmap="PuOr_r", vmin=-lim, vmax=lim, shading="nearest",
                        rasterized=True)
    cs = axd.contour(C.MAP_X, C.MAP_T, 0.5 * (p1 + p2), levels=[0.1, 0.5, 0.9], colors=st.INK, linewidths=0.5)
    axd.clabel(cs, fontsize=5.5, fmt="%.1f")
    axd.plot([C.X_DEC], [C.T_DEC], "+", color=st.INK, ms=5, mew=0.8)
    axd.set_xlabel(r"$x_\mathrm{Ni}$")
    axd.set_ylabel("Temperature (K)")
    cb = fig.colorbar(im, ax=axd, shrink=0.9, pad=0.01)
    cb.set_label("P(fcc), run 1 − run 2", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    se_d = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    summ["per_run_fcc_map"] = {"n_draws": [int(n1), int(n2)], "max_abs_diff": float(np.abs(p1 - p2).max()),
                               "mean_abs_diff": float(np.abs(p1 - p2).mean()),
                               "frac_cells_diff_gt_2se_binomial": float(np.mean(np.abs(p1 - p2) > 2 * se_d)),
                               "P_fcc_decision": [float(np.mean(e["joint_region"][run == k] == "FCC_A1"))
                                                  for k in (0, 1)]}
    st.panel_label(axd, "d")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(COL, LAB)], loc="outside lower center", ncol=3,
               fontsize=6.5)
    fig.savefig(C.FIG / "fig_mode_trace.pdf")
    fig.savefig(C.FIG / "fig_mode_trace.png", dpi=300)
    (C.DATA / "mode_trace_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
