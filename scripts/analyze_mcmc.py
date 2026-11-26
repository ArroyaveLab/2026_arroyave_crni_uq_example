#!/usr/bin/env python3
"""Step 4 diagnostics for two independent emcee runs. Env: calphad-uq (arviz 0.23.4).

PRE-REGISTERED RULE (written before the runs, REPORT.md step 4):
  tau      = integrated autocorrelation time per parameter (walker-averaged ACF, Sokal window c=5),
             estimated on the full chain of each run; tau_max = max over parameters.
  burn-in  = first ceil(5 * tau_max) steps of each run (tau_max of that run).
  converged when, for BOTH runs: N >= 50 * tau_max; and on the post-burn-in samples of both runs
  pooled (walkers as chains): rank-normalised split R-hat < 1.01 and bulk ESS >= 400 for every
  parameter; and the two runs' posterior means agree within 0.1 posterior sd.
usage: analyze_mcmc.py <run1_dir> <run2_dir> <out_json> [fig_prefix]
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni


def acf_mean(x):
    """x: (steps, walkers) -> walker-averaged normalised ACF."""
    n = x.shape[0]
    f = 2 ** int(np.ceil(np.log2(2 * n)))
    y = x - x.mean(axis=0)
    F = np.fft.rfft(y, n=f, axis=0)
    ac = np.fft.irfft(F * np.conj(F), axis=0)[:n]
    ac = ac.mean(axis=1)
    return ac / ac[0]


def tau_int(x, c=5.0):
    rho = acf_mean(x)
    taus = 2.0 * np.cumsum(rho) - 1.0
    m = np.arange(len(taus)) < c * taus
    w = np.argmin(m) if np.any(~m) else len(taus) - 1
    return float(taus[w])


def load(d):
    ch = np.load(Path(d) / "chain.npy")  # (steps, walkers, dim)
    lp = np.load(Path(d) / "lnprob.npy")
    return ch, lp


def main():
    import arviz as az
    r1, r2, outp = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    prefix = sys.argv[4] if len(sys.argv) > 4 else None
    runs = [load(r1), load(r2)]
    n = min(len(runs[0][0]), len(runs[1][0]))
    rep = {"rule": "burn=ceil(5 tau_max); N>=50 tau_max; split-Rhat<1.01; bulk ESS>=400; means agree <0.1 sd",
           "runs": []}
    post = []
    for (ch, lp), d in zip(runs, (r1, r2)):
        ch, lp = ch[:n], lp[:n]
        taus = [tau_int(ch[:, :, k]) for k in range(ch.shape[2])]
        tau_lp = tau_int(lp)
        tmax = max(taus)
        burn = int(np.ceil(5 * tmax))
        rep["runs"].append({"dir": d, "steps": int(n), "walkers": int(ch.shape[1]), "tau": taus,
                            "tau_lnprob": tau_lp, "tau_max": tmax, "burn": burn,
                            "N_over_tau_max": n / tmax, "acceptance": float(np.mean(np.any(np.diff(ch, axis=0) != 0, axis=2))),
                            "ess_N_W_over_tau": [(n - burn) * ch.shape[1] / t for t in taus],
                            "lnprob_max": float(lp.max())})
        post.append(ch[burn:])
    m = min(len(p) for p in post)
    pooled = np.concatenate([p[-m:] for p in post], axis=1)  # (steps, 2W, dim)
    idata = az.from_dict(posterior={lab: pooled[:, :, k].T for k, lab in enumerate(crni.LABELS)})
    rhat = az.rhat(idata, method="rank")
    ess_b = az.ess(idata, method="bulk")
    ess_t = az.ess(idata, method="tail")
    rep["pooled_post_burn_steps_per_run"] = int(m)
    rep["rhat"] = {k: float(rhat[k]) for k in crni.LABELS}
    rep["ess_bulk"] = {k: float(ess_b[k]) for k in crni.LABELS}
    rep["ess_tail"] = {k: float(ess_t[k]) for k in crni.LABELS}
    mu = [p.reshape(-1, p.shape[2]).mean(0) for p in post]
    sd = pooled.reshape(-1, pooled.shape[2]).std(0)
    rep["mean_diff_over_sd"] = (np.abs(mu[0] - mu[1]) / sd).tolist()
    rep["post_mean"] = pooled.reshape(-1, pooled.shape[2]).mean(0).tolist()
    rep["post_sd"] = sd.tolist()
    rep["converged"] = bool(all(r["N_over_tau_max"] >= 50 for r in rep["runs"])
                            and max(rep["rhat"].values()) < 1.01 and min(rep["ess_bulk"].values()) >= 400
                            and max(rep["mean_diff_over_sd"]) < 0.1)
    outp.write_text(json.dumps(rep, indent=1))
    print(json.dumps({"converged": rep["converged"], "steps": n,
                      "tau_max": [r["tau_max"] for r in rep["runs"]], "burn": [r["burn"] for r in rep["runs"]],
                      "N/tau": [round(r["N_over_tau_max"], 1) for r in rep["runs"]],
                      "rhat_max": max(rep["rhat"].values()), "ess_bulk_min": min(rep["ess_bulk"].values()),
                      "mean_diff_max": max(rep["mean_diff_over_sd"]),
                      "acc": [round(r["acceptance"], 3) for r in rep["runs"]]}))
    if prefix:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(6, 2, figsize=(10, 12), sharex=True)
        for k in range(10):
            ax = axs.flat[k]
            for (ch, lp), c in zip(runs, ("C0", "C3")):
                ax.plot(ch[:n, :, k], color=c, lw=0.3, alpha=0.35)
            ax.set_ylabel(crni.LABELS[k], fontsize=8)
        for j, ((ch, lp), c) in enumerate(zip(runs, ("C0", "C3"))):
            axs.flat[10 + j].plot(lp[:n], color=c, lw=0.3, alpha=0.35)
            axs.flat[10 + j].set_ylabel(f"lnprob run{j + 1}", fontsize=8)
            axs.flat[10 + j].set_ylim(np.percentile(lp[n // 5:], 0.5) - 5, lp.max() + 1)
            axs.flat[10 + j].axvline(rep["runs"][j]["burn"], color="k", ls="--")
        for ax in axs.flat:
            ax.tick_params(labelsize=7)
        fig.tight_layout()
        fig.savefig(prefix + "_traces.png", dpi=120)


if __name__ == "__main__":
    main()
