#!/usr/bin/env python3
"""Step 5: forward propagation of the converged posterior (hull phase diagram per draw).

Joint ensemble: post-burn-in samples of both runs pooled, thinned evenly to NDRAW draws.
Independent ensemble: the same draws with every parameter column permuted independently
(identical marginals, correlations destroyed).
For each draw: topology check over 800-2200 K, BCC/(BCC+FCC) and (BCC+FCC)/FCC boundaries,
liquidus on both sides, and the phase region at x_Ni = 0.52, 1473 K.
usage: propagate.py <diag_json> <set> <ndraw> <seed> <out_prefix>
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni
import pdfast
import plotpd

TT = np.arange(800.0, 2201.0, 5.0)


def evaluate(draws, ref, ref1473):
    tops, bds, at = [], [], []
    for th in draws:
        tops.append(pdfast.topology(th, TT, ref))
        bds.append(pdfast.boundaries(th, TT, ref))
        at.append(pdfast.phase_at(th, 1473.0, 0.52, ref1473))
    B = {k: np.array([b[k] for b in bds]) for k in bds[0]}
    return tops, B, at


def bands(B):
    out = {}
    for k, a in B.items():
        out[k] = {"q05": np.nanpercentile(a, 5, axis=0).tolist(), "q50": np.nanpercentile(a, 50, axis=0).tolist(),
                  "q95": np.nanpercentile(a, 95, axis=0).tolist(), "frac_present": np.mean(np.isfinite(a), axis=0).tolist()}
    return out


def main():
    diag = json.loads(Path(sys.argv[1]).read_text())
    setname, ndraw, seed, prefix = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    rng = np.random.default_rng(seed)
    post = []
    for r in diag["runs"]:
        ch = np.load(Path(r["dir"]) / "chain.npy")[:diag["runs"][0]["steps"]]
        post.append(ch[r["burn"]:].reshape(-1, ch.shape[2]))
    allp = np.concatenate(post)
    idx = np.linspace(0, len(allp) - 1, ndraw).astype(int)
    joint = allp[idx]
    indep = joint.copy()
    for k in range(indep.shape[1]):
        indep[:, k] = indep[rng.permutation(ndraw), k]
    ref = pdfast.ref_grid(TT, crni.TDB)
    ref1473 = pdfast.ref_grid([1473.0], crni.TDB)
    res = {}
    for name, D in (("joint", joint), ("independent", indep)):
        tops, B, at = evaluate(D, ref, ref1473)
        ok = np.array([t["ok"] for t in tops])
        fcc = np.array([a == "FCC_A1" for a in at])
        n = len(D)
        p = fcc.mean()
        res[name] = {"n": n, "P_topology_ok": float(ok.mean()),
                     "P_FCC_single_x052_1473K": float(p), "mc_se": float(np.sqrt(p * (1 - p) / n)),
                     "P_FCC_given_topology_ok": float(fcc[ok].mean()) if ok.any() else None,
                     "regions_at_point": {a: at.count(a) / n for a in sorted(set(at))},
                     "failure_examples": [t for t in tops if not t["ok"]][:5],
                     "T_liquid_min_q": np.percentile([t["T_liquid_min"] for t in tops], [5, 50, 95]).tolist(),
                     "bands": bands(B)}
        np.save(prefix + f"_{name}_draws.npy", D)
        print(json.dumps({k: res[name][k] for k in ("n", "P_topology_ok", "P_FCC_single_x052_1473K", "mc_se",
                                                      "P_FCC_given_topology_ok", "regions_at_point", "T_liquid_min_q")}))
    res["T"] = TT.tolist()
    Path(prefix + "_results.json").write_text(json.dumps(res, indent=1))
    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 5.6), sharey=True)
    theta_map = np.asarray(json.loads((crni.RUN / "fit" / f"curvature_{setname}.json").read_text())["theta_map"])
    for ax, name in zip(axs, ("joint", "independent")):
        bd = res[name]["bands"]
        for k, c in (("bcc_bf", "#1f6fb4"), ("fcc_bf", "#d1495b"), ("liq_bccside", "#2a9d55"),
                     ("liq_fccside", "#2a9d55"), ("sol_bcc", "#1f6fb4"), ("sol_fcc", "#d1495b")):
            lo, hi = np.array(bd[k]["q05"]), np.array(bd[k]["q95"])
            ax.fill_betweenx(TT, lo, hi, color=c, alpha=0.30, lw=0)
            ax.plot(bd[k]["q50"], TT, color=c, lw=0.8)
        plotpd.draw_boundaries(ax, theta_map, pdfast.ref_grid(plotpd.T, crni.TDB), color="k", lw=0.6, alpha=0.7)
        h = plotpd.draw_data(ax)
        ax.plot([0.52], [1473], "k+", ms=14, mew=2)
        r = res[name]
        ax.set_title(f"{name} ({r['n']} draws): 90% bands, MAP in black\n"
                     f"P(topology ok)={r['P_topology_ok']:.3f}; P(FCC at x=0.52, 1473 K)={r['P_FCC_single_x052_1473K']:.3f}",
                     fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(800, 2200); ax.set_xlabel(r"$x_\mathrm{Ni}$")
    axs[0].set_ylabel("T (K)")
    axs[1].legend(handles=h, fontsize=6, loc="upper right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(prefix + "_bands.png", dpi=160)
    # zoom near the decision point
    for ax in axs:
        ax.set_xlim(0.3, 0.75); ax.set_ylim(1300, 1700)
    fig.savefig(prefix + "_bands_zoom.png", dpi=160)


if __name__ == "__main__":
    main()
