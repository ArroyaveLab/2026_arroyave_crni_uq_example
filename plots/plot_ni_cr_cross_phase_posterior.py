"""Estimate projected Ni-Cr posterior densities for BCC-FCC coefficient pairs."""
from pathlib import Path

import matplotlib
import warnings

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chapter/data/ensembles.npz"
OUT = ROOT / "chapter/figures/fig_27_ni_cr_cross_phase_posterior"

data = np.load(DATA)
joint = data["joint_theta"]
independent = data["indep_theta"]
assert joint.shape == independent.shape == (2000, 10)

# Parameter order follows the Section 7 model: L0liq, L0bcc, L1bcc,
# L0fcc, L1fcc, with (a,b) coefficients for each interaction.
pairs = [("$a$ coefficients", 2, 6), ("$b$ coefficients", 3, 7)]
for _, ibcc, ifcc in pairs:
    assert np.allclose(np.sort(joint[:, ibcc]), np.sort(independent[:, ibcc]))
    assert np.allclose(np.sort(joint[:, ifcc]), np.sort(independent[:, ifcc]))


def density_and_hpd(samples, xx, yy):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                module=r"scipy\.stats\._kde")
        kde = gaussian_kde(samples.T, weights=np.full(len(samples), 1 / len(samples)))
    z = kde(np.vstack((xx.ravel(), yy.ravel()))).reshape(xx.shape)
    cell_area = (xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0])
    ordered = np.sort(z.ravel())[::-1]
    cumulative_mass = np.cumsum(ordered) * cell_area
    thresholds = [ordered[np.searchsorted(cumulative_mass, mass)]
                  for mass in (0.95, 0.80, 0.50)]
    return z, thresholds


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.65,
})
fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.3), layout="constrained")
colors = {"joint": "#286089", "independent": "#b86822"}
for col, title in enumerate(("Joint posterior", "Independent permutations")):
    axes[0, col].set_title(title, fontsize=7.1, pad=4)

correlations = {}
for row, (label, ibcc, ifcc) in enumerate(pairs):
    center_x, scale_x = joint[:, ibcc].mean(), joint[:, ibcc].std(ddof=1)
    center_y, scale_y = joint[:, ifcc].mean(), joint[:, ifcc].std(ddof=1)
    limits = []
    standardized = {}
    for name, samples in (("joint", joint), ("independent", independent)):
        x = (samples[:, ibcc] - center_x) / scale_x
        y = (samples[:, ifcc] - center_y) / scale_y
        standardized[name] = (x, y)
        limits.extend((x, y))
        correlations[name, row] = np.corrcoef(samples[:, ibcc], samples[:, ifcc])[0, 1]
    lim = max(abs(np.quantile(np.concatenate(limits), 0.005)),
              abs(np.quantile(np.concatenate(limits), 0.995))) * 1.08
    grid = np.linspace(-lim, lim, 150)
    xx, yy = np.meshgrid(grid, grid)

    for col, name in enumerate(("joint", "independent")):
        ax = axes[row, col]
        x, y = standardized[name]
        z, thresholds = density_and_hpd(np.column_stack((x, y)), xx, yy)
        ax.contourf(xx, yy, z, levels=[0, *thresholds, z.max() * 1.001],
                    colors=["#e7eff4", "#b8cbd8", colors[name]], alpha=0.95)
        ax.contour(xx, yy, z, levels=sorted(thresholds), colors=[colors[name]],
                   linewidths=[0.55, 0.7, 0.85])
        ax.text(0.96, 0.94, rf"$r={correlations[name, row]:+.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                      "pad": 1.0})
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=5.3, length=2, pad=1.2)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel(r"$L_0^{\mathrm{FCC}}$ / SD", fontsize=5.9, labelpad=1)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel(r"$L_0^{\mathrm{BCC}}$ / SD", fontsize=5.9, labelpad=1)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.text(0.05, 0.94, label, transform=ax.transAxes, ha="left",
                    va="top", fontsize=5.8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                          "pad": 0.8})

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
plt.close(fig)
print(f"Wrote {OUT.with_suffix('.pdf')} and {OUT.with_suffix('.png')}")
print("BCC-FCC a correlation: "
      f"joint={correlations['joint', 0]:+.3f}, independent={correlations['independent', 0]:+.3f}")
print("BCC-FCC b correlation: "
      f"joint={correlations['joint', 1]:+.3f}, independent={correlations['independent', 1]:+.3f}")
