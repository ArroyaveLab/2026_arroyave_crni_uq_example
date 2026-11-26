"""Plot actual Ni-Cr posterior parameter pairs and their marginal permutations."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chapter/data/ensembles.npz"
OUT = ROOT / "chapter/figures/fig_28_ni_cr_parameter_dependence"

data = np.load(DATA)
joint = data["joint_theta"]
independent = data["indep_theta"]
assert joint.shape == independent.shape == (2000, 10)

# The model parameter order is L0liq (a,b), L0bcc (a,b), L1bcc (a,b),
# L0fcc (a,b), L1fcc (a,b); see the Section 7 Gibbs-energy model.
pairs = [("BCC", 2, 3), ("FCC", 6, 7)]
for phase, ia, ib in pairs:
    assert np.allclose(np.sort(joint[:, ia]), np.sort(independent[:, ia]))
    assert np.allclose(np.sort(joint[:, ib]), np.sort(independent[:, ib]))

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7,
    "pdf.fonttype": 42,
    "axes.linewidth": 0.65,
})
fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.3), layout="constrained")
colors = {"joint": "#286089", "independent": "#b86822"}
column_titles = ("Joint posterior", "Independent permutations")
for col, title in enumerate(column_titles):
    axes[0, col].set_title(title, fontsize=7.1, pad=5)

for row, (phase, ia, ib) in enumerate(pairs):
    means = joint[:, [ia, ib]].mean(axis=0)
    scales = joint[:, [ia, ib]].std(axis=0, ddof=1)
    assert np.all(scales > 0)
    for col, (samples, color) in enumerate(
        ((joint, colors["joint"]), (independent, colors["independent"]))
    ):
        ax = axes[row, col]
        x = (samples[:, ia] - means[0]) / scales[0]
        y = (samples[:, ib] - means[1]) / scales[1]
        corr = np.corrcoef(samples[:, ia], samples[:, ib])[0, 1]
        ax.scatter(x, y, s=3.0, color=color, alpha=0.22, linewidths=0,
                   rasterized=True)
        ax.text(0.96, 0.94, rf"$r={corr:.3f}$", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.3,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                      "pad": 1.2})
        ax.text(0.06, 0.92, rf"$L_0^{{\mathrm{{{phase}}}}}$",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.3)
        limit = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.04
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect("equal", adjustable="box")
        ax.axhline(0, color="#a8afb3", lw=0.45, zorder=0)
        ax.axvline(0, color="#a8afb3", lw=0.45, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        if col == 0:
            ax.set_ylabel(r"$b/\mathrm{SD}(b)$", fontsize=6.2, labelpad=1)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel(r"$a/\mathrm{SD}(a)$", fontsize=6.2, labelpad=1)
        else:
            ax.set_xticklabels([])

fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=300)
plt.close(fig)
print(f"Wrote {OUT.with_suffix('.pdf')} and {OUT.with_suffix('.png')}")
for phase, ia, ib in pairs:
    print(f"{phase} L0 correlation: joint={np.corrcoef(joint[:, ia], joint[:, ib])[0, 1]:.3f}, "
          f"independent={np.corrcoef(independent[:, ia], independent[:, ib])[0, 1]:.3f}")
