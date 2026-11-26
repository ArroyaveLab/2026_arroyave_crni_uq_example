"""Check: ESPEI residuals of the single-phase BCC observations at the two set E fitted points."""
import sys, json, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts"); import numpy as np, crni
from espei.datasets import load_datasets
from espei.error_functions.context import setup_context
from pycalphad import Database
th = {n: np.asarray(json.load(open(f"fit/setE_{n}_theta.json"))["theta"]) for n in ("gapmode", "topook")}
for fn in ["zpf_single_CR-NI-ZPF-BCC_A2-FCC_A1-LIQUID-bechtoldt1961redetermination.json",
           "zpf_single_CR-NI-ZPF-BCC_A2-FCC_A1-dench1963adiabatic.json",
           "zpf_single_CR-NI-ZPF-BCC_A2-FCC_A1-jenkins1937some_alloys.json"]:
    f = crni.datasets_dir("setE_orchid_sigmas") / fn
    d = json.load(open(f)); ds = load_datasets([str(f)])
    ctx = setup_context(Database(str(crni.TDB)), ds, symbols_to_fit=crni.NAMES, phase_models=crni.phase_models())
    r = [o for o in ctx["residual_objs"] if type(o).__name__ == "ZPFResidual"][0]
    for n, t in th.items():
        df, w = r.get_residuals(t)
        df = np.asarray(df)
        rows = [(T, v[0][0], round(v[0][2][0], 3), round(float(x), 1)) for T, v, x in zip(d["conditions"]["T"], d["values"], df)]
        bcc_in = [z for z in rows if z[1] == "BCC_A2" and 0.1 <= z[2] <= 0.43 and 1365 <= z[0] <= 1640]
        print(fn[11:60], n, "BCC pts in gap region:", len(bcc_in), "driving forces J:", [z[3] for z in bcc_in][:14],
              "sum chi2/2:", round(float(np.sum((np.array([z[3] for z in bcc_in]) / 1000) ** 2) / 2), 2))
