"""Plot the run comparison, FCC-frequency maps and local Gaussian control.

All three figures are drawn from the archived arrays and JSON summaries of the
Cr-Ni example, without recomputing thermodynamics. Labels follow the wording
of Section 7 (first-fit field sequence, stable BCC+BCC field, retained joint
sample, modified local Gaussian control).
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chapter/data"
FIG = ROOT / "chapter/figures"

BLUE, ORANGE, PURPLE, GREY, INK = "#286089", "#b86822", "#6a3d9a", "#9a9a9a", "#222222"
CLASS_COLORS = [ORANGE, PURPLE, "#c8c8c8"]
CLASS_LABELS = ["First-fit field sequence", "Stable BCC+BCC field", "Other"]
X_DEC, T_DEC = 0.52, 1473.0
VALIDATED = (1200.0, 1550.0)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})


def panel(ax, letter):
    ax.text(-0.02, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            ha="right", va="bottom")


def save(fig, stem):
    fig.savefig(FIG / f"{stem}.pdf")
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    plt.close(fig)


def run_comparison():
    z = np.load(DATA / "mode_trace_K100.npz")
    e = np.load(DATA / "ensembles.npz")
    summary = json.loads((DATA / "mode_trace_summary.json").read_text())
    thin = int(z["K"])
    fig = plt.figure(figsize=(7.2, 4.5), layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    cmap = ListedColormap(CLASS_COLORS)
    axc = fig.add_subplot(gs[0, 1])
    starts = ["first", "second"]
    for r in (1, 2):
        cls, steps, burn = z[f"run{r}_cls"], z[f"run{r}_steps"], int(z[f"run{r}_burn"])
        assert burn == summary["runs"][f"run{r}"]["burn"]
        ax = fig.add_subplot(gs[r - 1, 0])
        ax.imshow(cls, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5, interpolation="nearest",
                  extent=[steps[0] - thin / 2, steps[-1] + thin / 2, cls.shape[0] + 0.5, 0.5])
        ax.axvline(burn, color=INK, lw=0.9, ls="--")
        ax.set(ylabel="Walker", xlabel="Step", yticks=[1, 10, 20])
        ax.set_title(f"Run started at the {starts[r - 1]} fitted point", fontsize=8, loc="left")
        ax.spines[["top", "right"]].set_visible(True)
        panel(ax, "ab"[r - 1])
        for k in range(3):
            axc.plot(steps, (cls == k).mean(0), color=CLASS_COLORS[k] if k < 2 else GREY, lw=0.8,
                     ls="-" if r == 1 else (0, (2.5, 1.2)))
    axc.set(xlabel="Step", ylabel="Fraction of walkers", ylim=(0, 1))
    axc.legend(handles=[Line2D([], [], color=INK, lw=0.8, label="First run"),
                        Line2D([], [], color=INK, lw=0.8, ls=(0, (2.5, 1.2)), label="Second run")],
               loc="center right", frameon=False, fontsize=7)
    panel(axc, "c")

    run = e["joint_meta"][:, 0]
    fmap = e["joint_fccmap"].astype(float)
    p1, p2 = fmap[run == 0].mean(0), fmap[run == 1].mean(0)
    assert ((run == 0).sum(), (run == 1).sum()) == (905, 1095)
    assert abs(np.abs(p1 - p2).max() - summary["per_run_fcc_map"]["max_abs_diff"]) < 1e-9
    axd = fig.add_subplot(gs[1, 1])
    im = axd.pcolormesh(e["MAP_X"], e["MAP_T"], p1 - p2, cmap="PuOr_r", vmin=-0.3, vmax=0.3,
                        shading="nearest", rasterized=True)
    cs = axd.contour(e["MAP_X"], e["MAP_T"], 0.5 * (p1 + p2), levels=[0.1, 0.5, 0.9], colors=INK,
                     linewidths=0.6)
    axd.clabel(cs, fontsize=6, fmt="%.1f")
    axd.plot([X_DEC], [T_DEC], "+", color=INK, ms=7, mew=1.0)
    axd.set(xlabel=r"$x_\mathrm{Ni}$", ylabel="Temperature (K)")
    cb = fig.colorbar(im, ax=axd, shrink=0.9, pad=0.02)
    cb.set_label("FCC frequency, first run $-$ second run", fontsize=7)
    panel(axd, "d")
    fig.legend(handles=[Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_LABELS)],
               loc="outside lower center", ncol=3, frameon=False, fontsize=7.5)
    save(fig, "fig_29_example_run_comparison")


def frequency_maps():
    m = np.load(DATA / "fcc_maps.npz")
    numbers = json.loads((DATA / "chapter_numbers.json").read_text())
    assert abs(numbers["maps"]["max_abs_joint_minus_indep"] - np.abs(m["joint"] - m["indep"]).max()) < 1e-9
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained")
    maps = [("joint", "Retained joint sample"), ("indep", "Independent marginals"),
            ("accepted", "Joint, first-fit sequence only")]
    for k, (key, title) in enumerate(maps):
        a = axs[0, k]
        im = a.pcolormesh(m["MAP_X"], m["MAP_T"], m[key], cmap="Oranges", vmin=0, vmax=1,
                          shading="nearest", rasterized=True)
        cs = a.contour(m["MAP_X"], m["MAP_T"], m[key], levels=[0.1, 0.5, 0.9], colors=INK, linewidths=0.6)
        a.clabel(cs, fontsize=6, fmt="%.1f")
        a.set_title(title, fontsize=8, loc="left")
        panel(a, "abc"[k])
    cb = fig.colorbar(im, ax=axs[0, 2], shrink=0.9, pad=0.02)
    cb.set_label("FCC frequency", fontsize=7)
    diffs = [("indep", "Joint $-$ independent"), ("accepted", "Joint $-$ first-fit sequence only")]
    for k, (key, title) in enumerate(diffs):
        a = axs[1, k]
        im2 = a.pcolormesh(m["MAP_X"], m["MAP_T"], m["joint"] - m[key], cmap="PuOr_r", vmin=-0.6, vmax=0.6,
                           shading="nearest", rasterized=True)
        a.contour(m["MAP_X"], m["MAP_T"], m["joint"], levels=[0.5], colors=INK, linewidths=0.6)
        a.set_title(title, fontsize=8, loc="left")
        panel(a, "de"[k])
    cb2 = fig.colorbar(im2, ax=axs[1, 1], shrink=0.9, pad=0.02)
    cb2.set_label("Difference in FCC frequency", fontsize=7)
    for a in axs.flat[:5]:
        a.plot([X_DEC], [T_DEC], "+", color=INK, ms=7, mew=1.0)
        a.set(xlabel=r"$x_\mathrm{Ni}$", ylabel="Temperature (K)")
        a.spines[["top", "right"]].set_visible(True)
    axs[1, 2].axis("off")
    save(fig, "fig_30_example_fcc_frequency_maps")


def local_gaussian():
    b = np.load(DATA / "boundary_bands.npz")
    s = json.loads((DATA / "laplace_summary.json").read_text())
    temperature = b["TT"]
    keep = (temperature >= VALIDATED[0]) & (temperature <= VALIDATED[1])
    t = temperature[keep]
    fig, (ax, a2) = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained",
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    for name, color, alpha in (("laplace", GREY, 0.55), ("accepted", ORANGE, 0.40)):
        for side in ("bcc_bf", "fcc_bf"):
            q = b[f"{name}_{side}"][:, keep]
            ax.fill_betweenx(t, q[0], q[2], color=color, alpha=alpha, lw=0)
    for side in ("bcc_bf", "fcc_bf"):
        q = b[f"joint_{side}"][:, keep]
        for k in (0, 2):
            ax.plot(q[k], t, color=PURPLE, lw=0.9, ls=(0, (3, 1.5)))
    ax.plot([X_DEC], [T_DEC], "+", color=INK, ms=7, mew=1.0)
    ax.set(xlim=(0, 0.85), ylim=VALIDATED, xlabel=r"$x_\mathrm{Ni}$", ylabel="Temperature (K)")
    ax.legend(handles=[Patch(color=GREY, alpha=0.75, label="Modified local Gaussian control"),
                       Patch(color=ORANGE, alpha=0.6, label="Joint sample, first-fit sequence only"),
                       Line2D([], [], color=PURPLE, lw=0.9, ls=(0, (3, 1.5)), label="Retained joint sample")],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=7)
    panel(ax, "a")
    labels = [r"liquid $a_0$", r"liquid $b_0$", r"BCC $a_0$", r"BCC $b_0$", r"BCC $a_1$", r"BCC $b_1$",
              r"FCC $a_0$", r"FCC $b_0$", r"FCC $a_1$", r"FCC $b_1$"]
    ratio = np.array(s["ratio_mcmc_over_laplace"])
    assert round(ratio.min(), 2) == 0.92 and round(ratio.max(), 2) == 1.82
    y = np.arange(10)[::-1]
    a2.barh(y, ratio, color=BLUE, height=0.6)
    a2.axvline(1, color=INK, lw=0.6)
    a2.set(xlim=(0, 2), yticks=y, yticklabels=labels,
           xlabel="Standard deviation,\njoint sample / Gaussian control")
    panel(a2, "b")
    save(fig, "fig_31_example_local_gaussian")


if __name__ == "__main__":
    run_comparison()
    frequency_maps()
    local_gaussian()
    print("Created three figures from archived arrays.")
