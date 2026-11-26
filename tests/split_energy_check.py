"""Check: energy of the BCC+BCC split relative to the stable hull at the observed single-phase BCC states."""
import sys, json, warnings; warnings.filterwarnings("ignore"); sys.path.insert(0, "scripts")
import numpy as np, crni, pdfast
st = []
for f in sorted(crni.datasets_dir("setE_orchid_sigmas").glob("zpf_single_*.json")):
    d = json.load(open(f))
    for T, r in zip(d["conditions"]["T"], d["values"]):
        for v in r:
            if v[0] == "BCC_A2":
                st.append((float(T), v[2][0] if v[1] == ["NI"] else 1 - v[2][0]))
st.append((1473.0, 0.282))
Ts = np.array(sorted({T for T, _ in st}))
ref = pdfast.ref_grid(Ts, crni.TDB)
out = {}
for n in ("topook", "gapmode"):
    th = np.asarray(json.load(open(f"fit/setE_{n}_theta.json"))["theta"])
    G = ref + pdfast.excess(th, Ts)
    D = []
    for T, x in st:
        i = int(np.where(Ts == T)[0][0]); g = G[0, i]  # BCC only
        h = pdfast._lower_hull(pdfast.X, g)
        hx, hg = pdfast.X[h], g[h]
        D.append(float(np.interp(x, pdfast.X, g) - np.interp(x, hx, hg)))
    D = np.array(D)
    out[n] = {"n_split_gt_1J": int((D > 1).sum()), "max_split_J": float(D.max()), "sum_half_sq_over_1000": float(0.5 * np.sum((D / 1000) ** 2)),
              "worst": [(st[k][0], round(st[k][1], 3), round(D[k], 1)) for k in np.argsort(-D)[:5]]}
print(json.dumps(out, indent=1)); json.dump(out, open("results/bcc_split_energy_check.json", "w"), indent=1)
