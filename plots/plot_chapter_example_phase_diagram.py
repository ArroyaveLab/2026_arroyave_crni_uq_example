"""Plot the topology-conditioned Cr-Ni phase diagram and boundary bands."""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "chapter/data"
OUT = ROOT / "chapter/figures/fig_27_example_phase_diagram"
WINDOW = (1350.0, 1550.0)
X_WINDOW = (0.0, 0.7)

bands = np.load(DATA / "boundary_bands.npz")
reference = json.loads((DATA / "reference_topology_1200_1550.json").read_text())
temperature = bands["TT"]
keep = (temperature >= WINDOW[0]) & (temperature <= WINDOW[1])
t = temperature[keep]

# Validate the reference graph and the retained posterior boundary summaries.
required_fields = ["BCC_A2", "BCC_A2+FCC_A1", "FCC_A1"]
assert reference["graph"]["fields"] == required_fields
assert reference["graph"]["invariant_reactions"] == []
assert reference["graph"]["terminal_conditions"] == {
    "x_NI->0": "BCC_A2", "x_NI->1": "FCC_A1"
}
ref_slices = [s for s in reference["slices"] if WINDOW[0] <= s["temperature_K"] <= WINDOW[1]]
assert len(ref_slices) >= 8
assert all(s["field_sequence"] == required_fields for s in ref_slices)
assert all(s["bcc_side_x_NI"] < s["fcc_side_x_NI"] for s in ref_slices)

bcc = bands["accepted_bcc_bf"][:, keep]
fcc = bands["accepted_fcc_bf"][:, keep]
for name, q in (("BCC boundary", bcc), ("FCC boundary", fcc)):
    assert np.isfinite(q).all(), f"Missing values in {name} bands"
    assert np.all(q[0] <= q[1]) and np.all(q[1] <= q[2]), f"Invalid {name} quantiles"
assert np.all(bcc[2] < fcc[0]), "The 5-95% boundary bands overlap or reverse field order"
assert np.all(bands["accepted_bcc_bf_present"][keep] == 1.0)
assert np.all(bands["accepted_fcc_bf_present"][keep] == 1.0)

ref_t = np.asarray([s["temperature_K"] for s in ref_slices])
ref_bcc = np.asarray([s["bcc_side_x_NI"] for s in ref_slices])
ref_fcc = np.asarray([s["fcc_side_x_NI"] for s in ref_slices])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(3.3, 2.8), layout="constrained")
blue, orange = "#286089", "#b86822"
ax.fill_betweenx(t, bcc[1], fcc[1], color="#d9e3e9", alpha=0.55, zorder=0)
ax.fill_betweenx(t, bcc[0], bcc[2], color=blue, alpha=0.20, linewidth=0,
                 label="5-95% boundary interval")
ax.fill_betweenx(t, fcc[0], fcc[2], color=orange, alpha=0.20, linewidth=0)
ax.plot(bcc[1], t, color=blue, lw=2.0, label="Posterior median boundary")
ax.plot(fcc[1], t, color=orange, lw=2.0)
ax.plot(ref_bcc, ref_t, color="#333333", lw=1.3, ls="--",
        label="Tang-Hallstedt reference")
ax.plot(ref_fcc, ref_t, color="#333333", lw=1.3, ls="--")
ax.scatter([0.52], [1473], marker="*", s=100, color="black",
           edgecolor="white", linewidth=0.6, zorder=5, label="Decision state")

ax.text(0.10, 1285, "BCC", ha="center", va="center", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2})
ax.text(0.39, 1285, "BCC + FCC", ha="center", va="center", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2})
ax.text(0.62, 1285, "FCC", ha="center", va="center", fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2})
ax.set(xlim=X_WINDOW, ylim=WINDOW,
       xlabel=r"Ni mole fraction, $x_{\mathrm{Ni}}$", ylabel="Temperature (K)")
ax.set_xticks(np.arange(0.0, 0.701, 0.1))
ax.set_yticks(np.arange(1350, 1551, 50))
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=5.8,
          ncol=2, handlelength=1.4, columnspacing=0.8, borderpad=0.3)
fig.savefig(OUT.with_suffix(".pdf"))
fig.savefig(OUT.with_suffix(".png"), dpi=220)
plt.close(fig)
print(f"Validated {len(t)} posterior temperature nodes and {len(ref_t)} reference slices")
print(f"Wrote {OUT.with_suffix('.pdf')} and {OUT.with_suffix('.png')}")
