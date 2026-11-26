#!/usr/bin/env python3
"""Decision probabilities, decision map and measurement ranking from data/ensembles.npz (no new equilibrium calls).

Decision: is x_Ni = 0.52 single-phase FCC_A1 at 1473 K (and, for the map, every (T, x) cell of MAP_T x MAP_X).
Monte Carlo errors: binomial; for the joint ensemble also a cluster (walker) standard error, because thinned MCMC
draws of one walker are correlated; for the accepted-conditioned probability a Wilson interval (z = 1).

Measurement ranking (posterior reweighting with the precomputed response tensor R[draw, candidate]):
  for a candidate c with noise sigma_c, every joint draw j in turn is taken as the truth, y = R[j,c] + sigma_c * eps;
  the joint draws are reweighted, w_i ~ exp(-(y - R[i,c])^2 / (2 sigma_c^2)); the decision uncertainty after the
  measurement is U = mean over map cells of p(1-p) (p = weighted P(fcc)); value = U_now - E_j[U_after].
  Also: the same for the decision state alone, for the topology mode (P(accepted)), and the mean weight ESS.
  Draws without the measured boundary (NaN response) get zero weight; truths with NaN are skipped (counted).
sigma (ORCHID orchid-uncertainty-calibration 2026-10-24, projects/orchid-data-next/literature/UNCERTAINTY_CALIBRATION.md):
  boundary composition X_PHASE: 0.011 (method unknown, calibrated on 3014 pairs) and 0.005 (microanalysis, prior);
  activity on mu = RT ln a: 2900 J/mol (method unknown, calibrated on 9 pairs) and 500 J/mol (emf, prior).
usage: analyze_decisions.py  -> data/chapter_numbers.json
"""
import json

import numpy as np

import common as C

SIG = {"x": {"calibrated_unknown": 0.011, "microanalysis_prior": 0.005},
       "mu": {"calibrated_unknown": 2900.0, "emf_prior": 500.0}}


def pstat(ind, groups=None):
    ind = np.asarray(ind, float)
    n = len(ind)
    p = float(ind.mean()) if n else float("nan")
    out = {"n": n, "p": p, "se_binomial": float(np.sqrt(p * (1 - p) / n)) if n else None}
    lo, hi = C.wilson(ind.sum(), n) if n else (None, None)
    out["wilson68"] = [float(lo), float(hi)] if n else None
    if groups is not None and n:
        out["se_cluster_walker"] = C.cluster_se(ind, groups)
    return out


def value_of(R, sigma, F, okv, dec, rng):
    """expected reductions for one candidate column R (n,), noise sigma."""
    n = len(R)
    good = np.isfinite(R)
    truths = np.where(good)[0]
    y = R[truths] + sigma * rng.standard_normal(len(truths))
    Rg = np.where(good, R, np.nan)
    L = -0.5 * ((y[:, None] - Rg[None, :]) / sigma) ** 2
    L = np.where(np.isfinite(L), L, -np.inf)
    L -= L.max(1, keepdims=True)
    W = np.exp(L)
    W /= W.sum(1, keepdims=True)
    P = W @ F                      # (truths, cells)
    Umap = (P * (1 - P)).mean(1)
    pd = W @ dec
    po = W @ okv
    ess = 1 / (W ** 2).sum(1)
    p0, pd0, po0 = F.mean(0), dec.mean(), okv.mean()
    U0 = float((p0 * (1 - p0)).mean())
    red = U0 - Umap
    bs = [red[rng.integers(0, len(red), len(red))].mean() for _ in range(200)]
    return {"dU_map": float(red.mean()), "dU_map_se_boot": float(np.std(bs)), "U_map_now": U0,
            "dU_map_rel": float(red.mean() / U0),
            "dU_point": float(pd0 * (1 - pd0) - np.mean(pd * (1 - pd))),
            "dU_point_rel": float(1 - np.mean(pd * (1 - pd)) / (pd0 * (1 - pd0))),
            "dU_mode": float(po0 * (1 - po0) - np.mean(po * (1 - po))),
            "dU_mode_rel": float(1 - np.mean(po * (1 - po)) / (po0 * (1 - po0))),
            "mean_weight_ess": float(ess.mean()), "n_truths": int(len(truths)), "n_nan_responses": int(n - good.sum())}


def main():
    e = dict(np.load(C.DATA / "ensembles.npz"))
    # Cr-side (first) BCC+FCC tie-line replaces pdfast.boundaries' last one (compute_boundaries_first.py)
    B1 = np.load(C.DATA / "boundaries_first.npz")
    for nm in ("joint", "indep", "laplace"):
        e[f"{nm}_bcc_bf"], e[f"{nm}_fcc_bf"] = B1[f"{nm}_bcc_bf1"], B1[f"{nm}_fcc_bf1"]
        e[f"{nm}_second_bcc"] = B1[f"{nm}_second_bcc"]
    meta = json.loads((C.DATA / "ensembles_meta.json").read_text())
    for c in meta["candidates"]:  # boundary responses from the Cr-side tie-line
        if c["kind"] in ("x_bcc_boundary", "x_fcc_boundary"):
            k = "bcc_bf" if c["kind"] == "x_bcc_boundary" else "fcc_bf"
            e["joint_resp"][:, c["i"]] = e[f"joint_{k}"][:, int(np.argmin(abs(C.TT - c["T"])))]
    jm = e["joint_meta"]
    grp = jm[:, 0] * 100 + jm[:, 2]
    out = {"ensembles_meta": {k: meta[k] for k in ("N_joint", "N_laplace", "rebuild_400_identical", "seed_indep",
                                                   "seed_laplace", "curvature_file", "wall_s", "nproc")}}
    # ---- decision-state probabilities
    res = {}
    for name in ("joint", "indep", "laplace", "j400"):
        cls, reg = e[f"{name}_cls"], e[f"{name}_region"]
        fcc = reg == "FCC_A1"
        g = grp if name == "joint" else None
        r = {"P_ok": pstat(cls == 0, g), "P_gap": pstat(cls == 1, g), "P_other": pstat(cls == 2, g),
             "P_fcc": pstat(fcc, g),
             "P_fcc_given_ok": pstat(fcc[cls == 0], g[cls == 0] if g is not None else None),
             "P_fcc_given_gap": pstat(fcc[cls == 1], g[cls == 1] if g is not None else None),
             "P_fcc_given_other": pstat(fcc[cls == 2]),
             "regions": {k: float(np.mean(reg == k)) for k in sorted(set(reg.tolist()))}}
        if name != "j400":
            sb = e[f"{name}_second_bcc"]
            r["second_bcc_field_1200_1600K"] = {"all": float(sb.mean()),
                                                **{C.CLASSES[k]: float(sb[cls == k].mean()) if (cls == k).any() else None
                                                   for k in range(3)},
                                                "P_fcc_given_no_second": pstat(fcc[~sb]),
                                                "P_fcc_given_second": pstat(fcc[sb]),
                                                "P_fcc_given_gap_no_second": pstat(fcc[(cls == 1) & ~sb])}
        if name == "joint":
            r["per_run"] = {f"run{k + 1}": {"P_ok": pstat((cls == 0)[jm[:, 0] == k], grp[jm[:, 0] == k]),
                                            "P_fcc": pstat(fcc[jm[:, 0] == k], grp[jm[:, 0] == k])}
                            for k in (0, 1)}
        res[name] = r
    saved = json.loads((C.RUN / "results/notconverged/setE_prop400_results.json").read_text())
    res["j400_vs_saved"] = {"P_ok": [res["j400"]["P_ok"]["p"], saved["joint"]["P_topology_ok"]],
                            "P_fcc": [res["j400"]["P_fcc"]["p"], saved["joint"]["P_FCC_single_x052_1473K"]]}
    out["decision_state"] = res
    # ---- maps
    F = {k: e[f"{k}_fccmap"].astype(float) for k in ("joint", "indep", "laplace")}
    ok = e["joint_cls"] == 0
    maps = {"joint": F["joint"].mean(0), "indep": F["indep"].mean(0), "laplace": F["laplace"].mean(0),
            "accepted": F["joint"][ok].mean(0), "gap": F["joint"][e["joint_cls"] == 1].mean(0)}
    np.savez(C.DATA / "fcc_maps.npz", MAP_T=C.MAP_T, MAP_X=C.MAP_X, **maps)
    u = {k: float((m * (1 - m)).mean()) for k, m in maps.items()}
    band = {k: int(((m > 0.05) & (m < 0.95)).sum()) for k, m in maps.items()}
    out["maps"] = {"U_map_mean_p1mp": u, "n_cells_0.05_0.95": band, "n_cells": int(maps["joint"].size),
                   "max_abs_joint_minus_indep": float(np.abs(maps["joint"] - maps["indep"]).max()),
                   "max_abs_joint_minus_accepted": float(np.abs(maps["joint"] - maps["accepted"]).max()),
                   "max_abs_joint_minus_laplace": float(np.abs(maps["joint"] - maps["laplace"]).max())}
    # ---- boundary bands (5/50/95) for joint, accepted-conditioned, laplace
    bands = {}
    for name, sel in (("joint", slice(None)), ("accepted", ok), ("laplace", slice(None)), ("indep", slice(None))):
        src = "laplace" if name == "laplace" else ("indep" if name == "indep" else "joint")
        for b in ("bcc_bf", "fcc_bf"):
            a = e[f"{src}_{b}"][sel]
            bands[f"{name}_{b}"] = np.nanpercentile(a, [5, 50, 95], axis=0)
            bands[f"{name}_{b}_present"] = np.mean(np.isfinite(a), axis=0)
    np.savez(C.DATA / "boundary_bands.npz", TT=C.TT, **bands)
    i1473 = int(np.argmin(abs(C.TT - 1473)))
    out["bands_1473K"] = {k: bands[k][:, i1473].tolist() for k in bands if not k.endswith("present")}
    # ---- measurement ranking
    Rt = e["joint_resp"]
    okv = ok.astype(float)
    dec = (e["joint_region"] == "FCC_A1").astype(float)
    gp = e["joint_cls"] == 1
    rank = []
    rng = np.random.default_rng(20261108)
    for c in meta["candidates"]:
        kind = "x" if c["kind"].startswith("x_") else "mu"
        col = Rt[:, c["i"]]
        a, b = col[ok], col[gp]
        sepv = {}
        for sname, s in SIG[kind].items():
            sepv[sname] = value_of(col, s, F["joint"].reshape(len(col), -1), okv, dec, rng)
            sepv[sname]["separation_index"] = float(abs(np.nanmean(a) - np.nanmean(b)) /
                                                    np.sqrt(s ** 2 + 0.5 * (np.nanvar(a) + np.nanvar(b))))
        rank.append({**c, "mean_ok": float(np.nanmean(a)), "mean_gap": float(np.nanmean(b)),
                     "sd_ok": float(np.nanstd(a)), "sd_gap": float(np.nanstd(b)), "sd_all": float(np.nanstd(col)),
                     "by_sigma": sepv})
    out["sigma"] = SIG
    out["ranking"] = rank
    # rank correlation of value with mode separation (calibrated sigmas)
    from scipy.stats import spearmanr
    v = [r["by_sigma"]["calibrated_unknown"]["dU_map"] for r in rank]
    s = [r["by_sigma"]["calibrated_unknown"]["separation_index"] for r in rank]
    m = [r["by_sigma"]["calibrated_unknown"]["dU_mode"] for r in rank]
    out["ranking_check"] = {"spearman_value_vs_separation": float(spearmanr(v, s)[0]),
                            "spearman_value_vs_mode_reduction": float(spearmanr(v, m)[0]),
                            "top5_by_value": [f"{rank[i]['kind']} T={rank[i]['T']} x={rank[i]['x_Ni']}"
                                              for i in np.argsort(v)[::-1][:5]],
                            "top5_by_separation": [f"{rank[i]['kind']} T={rank[i]['T']} x={rank[i]['x_Ni']}"
                                                   for i in np.argsort(s)[::-1][:5]]}
    (C.DATA / "chapter_numbers.json").write_text(json.dumps(out, indent=1))
    d = out["decision_state"]
    for k in ("joint", "indep", "laplace", "j400"):
        print(k, {q: round(d[k][q]["p"], 3) for q in ("P_ok", "P_gap", "P_fcc", "P_fcc_given_ok", "P_fcc_given_gap")})
    print(json.dumps(out["ranking_check"], indent=1))


if __name__ == "__main__":
    main()
