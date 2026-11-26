"""Descriptive (not pre-registered): topology indicator per walker along each run, thinned every K steps after burn-in.
Gives P(topology ok) per run, mode crossings per walker, and the ESS of the indicator. usage: mode_crossings.py <diag_json> <K> <out_json>"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import crni, pdfast
TT = np.arange(800.0, 2201.0, 5.0)
diag = json.loads(Path(sys.argv[1]).read_text()); K = int(sys.argv[2])
ref = pdfast.ref_grid(TT, crni.TDB)
out = {}
for r in diag["runs"]:
    ch = np.load(Path(r["dir"]) / "chain.npy")
    steps = np.arange(r["burn"], ch.shape[0], K)
    ind = np.array([[pdfast.topology(ch[s, w], TT, ref)["ok"] for s in steps] for w in range(ch.shape[1])], float)
    cross = np.abs(np.diff(ind, axis=1)).sum(axis=1)
    x = ind - ind.mean()
    n = ind.shape[1]
    acf = np.array([np.mean([np.dot(xw[:n-k], xw[k:]) for xw in x]) for k in range(n // 2)])
    acf = acf / acf[0] if acf[0] > 0 else acf
    tau = 1 + 2 * np.sum(acf[1:np.argmax(acf < 0) if np.any(acf < 0) else len(acf)])
    out[r["dir"]] = {"K": K, "n_per_walker": int(n), "P_ok": float(ind.mean()), "P_ok_per_walker": ind.mean(1).round(3).tolist(),
                     "crossings_per_walker": cross.astype(int).tolist(), "total_crossings": int(cross.sum()),
                     "tau_indicator_in_K_steps": float(tau), "ess_indicator": float(ind.size / tau)}
    print(r["dir"], {k: out[r["dir"]][k] for k in ("P_ok", "total_crossings", "tau_indicator_in_K_steps", "ess_indicator")}, flush=True)
Path(sys.argv[3]).write_text(json.dumps(out, indent=1))
