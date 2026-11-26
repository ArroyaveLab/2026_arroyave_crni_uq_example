"""Check: critical temperature of the BCC miscibility gap from the DFT mixing enthalpies and from
the two set E fitted points."""
import sys, json; sys.path.insert(0, "scripts"); import numpy as np, crni, bcc_convexity as bc
# DFT A2 (Liu 2016): x_Cr .75/.5/.25 -> x_Ni .25/.5/.75 ; H = x(1-x)[L0 + L1(1-2x)]
x = np.array([0.25, 0.5, 0.75]); H = np.array([10.02, 9.15, 4.98]) * 1000
A = np.stack([x * (1 - x), x * (1 - x) * (1 - 2 * x)], 1); (L0, L1), *_ = np.linalg.lstsq(A, H, rcond=None)
out = {"DFT_fit_L0_L1_J": [L0, L1], "Tc_regular_DFT_K (4H(0.5)/2R)": 4 * 9150 / (2 * bc.R),
       "Tc_subregular_DFT_K": bc.tc_estimate(L0, 0, L1, 0)}
for n in ("topook", "gapmode"):
    th = np.asarray(json.load(open(f"fit/setE_{n}_theta.json"))["theta"])
    out[f"Tc_fit_{n}"] = bc.tc_estimate(th[2], th[3], th[4], th[5])
out["Tc_paper_opt"] = bc.tc_estimate(*crni.PAPER_OPT[2:6])
# observed single-phase BCC states (set E (i) files: all BCC known vertices of the ZPF data)
st = []
for f in sorted(crni.datasets_dir("setE_orchid_sigmas").glob("zpf_single_*.json")):
    d = json.load(open(f))
    for T, r in zip(d["conditions"]["T"], d["values"]):
        for v in r:
            if v[0] == "BCC_A2":
                st.append((T, v[2][0] if v[1] == ["NI"] else 1 - v[2][0], f.name))
zh = json.load(open(crni.datasets_dir("setE_orchid_sigmas") / "zpf_CR-NI-ZPF-BCC_A2-FCC_A1-zhang2014impurity.json"))
st.append((1473.0, 0.282, "zhang2014 (tie-line BCC end)"))
out["n_states"] = len(st)
pen = bc.ConvexityPenalty([(T, x) for T, x, _ in st])
for n, th in (("paper_opt", crni.PAPER_OPT), ("topook", np.asarray(json.load(open("fit/setE_topook_theta.json"))["theta"])),
              ("gapmode", np.asarray(json.load(open("fit/setE_gapmode_theta.json"))["theta"]))):
    P, G = pen(th); v = pen.violations(th)
    fd = np.zeros(10)
    for i in range(10):
        h = 1e-4 * max(1, abs(th[i])); a = th.copy(); b = th.copy(); a[i] += h; b[i] -= h
        fd[i] = (pen(a)[0] - pen(b)[0]) / (2 * h)
    out[n] = {"penalty": P, "n_violations": len(v), "min_curvature_J": min(bc.curvature(th, T, x) for T, x, _ in st),
              "grad_vs_fd_rel": float(np.max(np.abs(fd - G)) / (np.max(np.abs(G)) + 1e-30)) if P > 0 else None,
              "worst": sorted(v, key=lambda z: z[2])[:3]}
json.dump(out, open("results/bcc_convexity_check.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
