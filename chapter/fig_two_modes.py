#!/usr/bin/env python3
"""fig_two_modes: the accepted-mode and gap-mode optima of the set E posterior, computed (pdfast hull, 1 K steps),
with the set E ZPF data and the metastable BCC_A2 miscibility gap (binodal dashed, chemical spinodal dotted).
Top: full diagrams; bottom: zoom on the gap. Writes figures/fig_two_modes.{pdf,png} and data/fig_two_modes.json.
"""
import json

import numpy as np

import common as C
import pd_draw as D

st = C.style()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

T_FULL = np.arange(800.0, 2201.0, 1.0)
T_ZOOM = np.arange(1150.0, 1701.0, 1.0)
MODES = [("accepted", "Accepted mode (optimum)"), ("gap", "Gap mode (optimum)")]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(st.W2, 126 * st.MM), constrained_layout=True,
                            gridspec_kw={"height_ratios": [1, 1]})
    info = {}
    for j, (m, title) in enumerate(MODES):
        th = C.theta_of(m)
        lp = json.loads((C.RUN / "results" / ("setE_orchid_sigmas_best.json" if m == "accepted"
                                              else "setE_orchid_sigmas_gapmode.json")).read_text())["logpost"]
        ex = D.gap_excursion(th, T_ZOOM)
        info[m] = {"logpost": lp, **{k: v for k, v in ex.items() if k != "depth_by_T"}}
        for i, (ax, T) in enumerate(((axs[0, j], T_FULL), (axs[1, j], T_ZOOM))):
            D.draw(ax, th, T, st.C_MODEL, st=st, mcolor=st.INK2)
            ax.plot([C.X_DEC], [C.T_DEC], "+", color=st.INK, ms=5, mew=0.8, zorder=4)
            ax.set_xlabel(r"$x_\mathrm{Ni}$")
            ax.set_ylabel("Temperature (K)")
        axs[0, j].set_xlim(0, 1); axs[0, j].set_ylim(800, 2200)
        axs[1, j].set_xlim(0, 0.65); axs[1, j].set_ylim(1150, 1700)
        a = axs[0, j]
        a.text(0.08, 1100, "bcc", fontsize=6.5, color=st.INK2)
        a.text(0.78, 1100, "fcc", fontsize=6.5, color=st.INK2)
        a.text(0.35, 1000, "bcc + fcc", fontsize=6.5, color=st.INK2, ha="center")
        a.text(0.5, 2050, "liquid", fontsize=6.5, color=st.INK2, ha="center")
        a.text(0.98, 2160, f"{title}\nlog posterior {lp:.2f}", fontsize=6.5, ha="right", va="top")
        z = axs[1, j]
        bt = ex["binodal_top_K"]
        d = ex["max_signed_depth_J"]
        box = dict(boxstyle="square,pad=0.25", fc="white", ec="none")
        if ex["stable_gap_T"]:
            t0, t1 = ex["stable_gap_T"]
            z.annotate("", xy=(0.012, t1), xytext=(0.012, t0),
                       arrowprops=dict(arrowstyle="<->", lw=0.5, color=st.INK, shrinkA=0, shrinkB=0))
            txt = (f"bcc-only gap top {bt:.0f} K\nbcc + bcc stable {t0:.0f}–{t1:.0f} K (arrow)\n"
                   f"Ni-rich gap end up to {d[1]:.0f} J mol$^{{-1}}$\nbelow the bcc + fcc hull ({d[0]:.0f} K)")
        else:
            txt = (f"bcc-only gap top {bt:.0f} K\nhidden in bcc + fcc: Ni-rich gap\n"
                   f"end at least {-d[1]:.0f} J mol$^{{-1}}$ above the hull")
        z.text(0.01, 1695, txt, fontsize=6, va="top", bbox=box, zorder=5)
        z.axhline(info.get("accepted", {}).get("binodal_top_K", bt), color=st.GREY_M, lw=0.4, zorder=0)
    # 75 K arrow: gap-mode binodal top minus accepted-mode binodal top, drawn on the gap-mode zoom
    ta, tg = info["accepted"]["binodal_top_K"], info["gap"]["binodal_top_K"]
    z = axs[1, 1]
    z.annotate("", xy=(0.27, tg), xytext=(0.27, ta),
               arrowprops=dict(arrowstyle="<->", lw=0.6, color=st.ACCENT, shrinkA=0, shrinkB=0))
    z.text(0.28, 0.5 * (ta + tg), f"{tg - ta:.0f} K", color=st.ACCENT, fontsize=6.5, va="center",
           bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    info["gap_top_difference_K"] = tg - ta
    h = [Line2D([], [], color=st.C_MODEL, lw=0.8, label="Stable boundaries (computed)"),
         Line2D([], [], color=st.INK2, lw=0.7, ls="--", dashes=(3, 1.5), label="bcc-only binodal (metastable)"),
         Line2D([], [], color=st.INK2, lw=0.8, ls=":", label="bcc-only chemical spinodal"),
         Line2D([], [], ls="", marker="s", ms=2.6, mfc="none", mec=st.C_EXP, mew=0.5, label="ZPF data, bcc vertex"),
         Line2D([], [], ls="", marker="o", ms=2.6, mfc="none", mec=st.C_EXP, mew=0.5, label="fcc vertex"),
         Line2D([], [], ls="", marker="^", ms=2.6, mfc="none", mec=st.C_EXP, mew=0.5, label="liquid vertex"),
         Line2D([], [], ls="", marker="+", ms=5, color=st.INK, mew=0.8, label="Decision state (0.52, 1473 K)")]
    fig.legend(handles=h, loc="outside lower center", ncol=4, fontsize=6, handlelength=2.2)
    for ax, l in zip(axs.flat, "abcd"):
        st.panel_label(ax, l)
    fig.savefig(C.FIG / "fig_two_modes.pdf")
    fig.savefig(C.FIG / "fig_two_modes.png", dpi=300)
    (C.DATA / "fig_two_modes.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
