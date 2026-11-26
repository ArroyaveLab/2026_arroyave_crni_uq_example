#!/usr/bin/env python3
"""fig_measurements: candidate measurements ranked by expected reduction of decision-relevant uncertainty
(posterior reweighting of the joint draws with the precomputed response tensor; analyze_decisions.py).
(a) relative reduction of the map-averaged Bernoulli variance, ORCHID-calibrated sigma (filled) and the
    optimistic method sigma (open); (b) value against the mode-separation index; (c) predicted responses of the
    top-ranked candidate in the two topology modes.
Reads data/chapter_numbers.json and data/ensembles.npz.
"""
import json

import numpy as np

import common as C

st = C.style()
import matplotlib.pyplot as plt  # noqa: E402

KIND = {"x_bcc_boundary": ("bcc/(bcc+fcc) boundary", "s"), "x_fcc_boundary": ("(bcc+fcc)/fcc boundary", "o"),
        "mu_Cr_bcc_field": (r"$a_\mathrm{Cr}$, Cr-rich alloy", "D"), "mu_Cr_fcc_field": (r"$a_\mathrm{Cr}$, fcc alloy", "^")}
OPT = {"x": "microanalysis_prior", "mu": "emf_prior"}


def label(r):
    k = KIND[r["kind"]][0]
    return f"{k}, {r['T']:.0f} K" + (f", {r['x_Ni']:.2f}" if r["x_Ni"] is not None else "")


def main():
    N = json.loads((C.DATA / "chapter_numbers.json").read_text())
    R = N["ranking"]
    v = np.array([r["by_sigma"]["calibrated_unknown"]["dU_map_rel"] for r in R])
    order = np.argsort(v)[::-1]
    fig = plt.figure(figsize=(st.W2, 105 * st.MM), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1])
    ax = fig.add_subplot(gs[:, 0])
    y = np.arange(len(R))[::-1]
    for yi, i in zip(y, order):
        r = R[i]
        kind = "x" if r["kind"].startswith("x_") else "mu"
        a = r["by_sigma"]["calibrated_unknown"]
        b = r["by_sigma"][OPT[kind]]
        mk = KIND[r["kind"]][1]
        ax.plot([a["dU_map_rel"], b["dU_map_rel"]], [yi, yi], color=st.GREY_M, lw=0.6, zorder=1)
        ax.errorbar(a["dU_map_rel"], yi, xerr=a["dU_map_se_boot"] / a["U_map_now"], fmt=mk, ms=3,
                    color=st.C_MODEL, lw=0.5, zorder=2)
        ax.plot(b["dU_map_rel"], yi, mk, ms=3, mfc="white", mec=st.C_MODEL, mew=0.6, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([label(R[i]) for i in order], fontsize=5.5)
    ax.set_xlabel("Expected reduction of decision-map uncertainty (fraction)")
    ax.plot([], [], "o", color=st.C_MODEL, ms=3, label="ORCHID-calibrated σ (method unknown)")
    ax.plot([], [], "o", mfc="white", mec=st.C_MODEL, ms=3, label="Optimistic method σ (prior)")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlim(left=0)
    st.panel_label(ax, "a")
    b = fig.add_subplot(gs[0, 1])
    for r in R:
        a = r["by_sigma"]["calibrated_unknown"]
        b.plot(a["separation_index"], a["dU_map_rel"], KIND[r["kind"]][1], ms=3, mfc="none", mec=st.C_MODEL, mew=0.6)
    b.set_xlabel("Mode separation index")
    b.set_ylabel("Expected reduction (fraction)")
    rc = N["ranking_check"]
    b.text(0.03, 0.97, f"Spearman ρ = {rc['spearman_value_vs_separation']:.2f}", transform=b.transAxes, va="top",
           fontsize=6)
    st.panel_label(b, "b")
    c = fig.add_subplot(gs[1, 1])
    e = np.load(C.DATA / "ensembles.npz")
    top = R[order[0]]
    col = e["joint_resp"][:, top["i"]]
    if top["kind"].startswith("x_"):  # Cr-side tie-line, as in analyze_decisions.py
        B1 = np.load(C.DATA / "boundaries_first.npz")
        k = "bcc_bf1" if top["kind"] == "x_bcc_boundary" else "fcc_bf1"
        col = B1[f"joint_{k}"][:, int(np.argmin(abs(C.TT - top["T"])))]
    cls = e["joint_cls"]
    scale = 1.0 if top["kind"].startswith("x_") else 1e-3
    vals = col[np.isfinite(col)] * scale
    bins = np.linspace(vals.min(), vals.max(), 40)
    for k, colr, lab in ((0, st.C_MODEL, "accepted"), (1, st.ACCENT, "gap"), (2, st.GREY_M, "other")):
        sel = col[(cls == k) & np.isfinite(col)] * scale
        if len(sel):
            c.hist(sel, bins=bins, color=colr, alpha=0.6, label=f"{lab} (n={len(sel)})")
    kind = "x" if top["kind"].startswith("x_") else "mu"
    sig = N["sigma"][kind]["calibrated_unknown"] * scale
    c.errorbar(np.median(vals), c.get_ylim()[1] * 0.55, xerr=sig, color=st.INK, capsize=2, lw=0.7)
    c.text(np.median(vals), c.get_ylim()[1] * 0.58, "±σ", fontsize=6, ha="center", va="bottom")
    c.set_xlabel(label(top) + (r" ($x_\mathrm{Ni}$)" if kind == "x" else r" (RT ln a, kJ mol$^{-1}$)"), fontsize=6.5)
    c.set_ylabel("Joint draws")
    c.legend(fontsize=6, loc="upper right")
    st.panel_label(c, "c")
    fig.savefig(C.FIG / "fig_measurements.pdf")
    fig.savefig(C.FIG / "fig_measurements.png", dpi=300)


if __name__ == "__main__":
    main()
