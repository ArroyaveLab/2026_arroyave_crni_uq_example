#!/usr/bin/env python3
"""Plot the Cr-Ni phase diagram (hull method) for a parameter vector, with the ZPF data.

usage: plotpd.py <theta.json | paper_opt | paper_start> <out.png> [title]
theta.json: a JSON file with key "theta" (list of 10) or a bare list.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import crni
import pdfast

T = np.arange(800.0, 2201.0, 2.0)
MARK = {"bechtoldt1961redetermination": "o", "collins1988electron": "s", "dench1963adiabatic": "^",
        "jenkins1937some_alloys": "v", "jette1934xray": "x", "karmazin1982lattice": "D",
        "taylor1952constitution": "P", "zhang2014impurity": "*", "svechnikov1962characteristics": "<"}
PCOL = {"BCC_A2": "#1f6fb4", "FCC_A1": "#d1495b", "LIQUID": "#2a9d55"}


def load_theta(arg):
    if arg == "paper_opt":
        return crni.PAPER_OPT
    if arg == "paper_start":
        return crni.PAPER_START
    d = json.loads(Path(arg).read_text())
    return np.asarray(d["theta"] if isinstance(d, dict) else d, float)


def draw_boundaries(ax, theta, ref, color="k", lw=1.2, alpha=1.0, label=None):
    b = pdfast.boundaries(theta, T, ref)
    first = True
    for k in b:
        ax.plot(b[k], T, "-", color=color, lw=lw, alpha=alpha, label=label if first else None)
        first = False
    return b


def draw_data(ax):
    for f in sorted((crni.INP / "chapter-input-data").glob("zpf_*.json")):
        d = json.loads(f.read_text())
        Ts = d["conditions"]["T"]
        Ts = Ts if isinstance(Ts, list) else [Ts] * len(d["values"])
        ref = d["reference"]
        for t, reg in zip(Ts, d["values"]):
            for ph, _, (x,) in reg:
                if x is None:
                    continue
                xn = x if _ == ["NI"] else 1 - x
                ax.plot(xn, t, MARK.get(ref, "o"), ms=4.5 if ref != "zhang2014impurity" else 11,
                        mfc="none" if ref != "zhang2014impurity" else PCOL[ph], mec=PCOL[ph], mew=0.9)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker=m, mfc="none", mec="0.3", label=r.split("19")[0].split("20")[0].capitalize()
                + " " + "".join(c for c in r if c.isdigit())) for r, m in MARK.items()]
    h += [Line2D([], [], ls="", marker="o", mfc="none", mec=c, label=p + " vertex") for p, c in PCOL.items()]
    return h


def main():
    theta = load_theta(sys.argv[1])
    out = Path(sys.argv[2])
    title = sys.argv[3] if len(sys.argv) > 3 else ""
    ref = pdfast.ref_grid(T, crni.TDB)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    draw_boundaries(ax, theta, ref)
    h = draw_data(ax)
    top = pdfast.topology(theta, np.arange(800.0, 2201.0, 5.0), pdfast.ref_grid(np.arange(800.0, 2201.0, 5.0), crni.TDB))
    ax.set_xlim(0, 1); ax.set_ylim(800, 2200)
    ax.set_xlabel(r"$x_\mathrm{Ni}$"); ax.set_ylabel("T (K)")
    ax.set_title((title + "\n" if title else "") + f"topology ok={top['ok']}, lowest liquid T={top['T_liquid_min']:.0f} K",
                 fontsize=9)
    ax.axhline(1618, color="0.7", lw=0.6, ls=":")
    ax.legend(handles=h, fontsize=6.5, loc="upper right", ncol=2, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print(json.dumps({"out": str(out), "topology": top}))


if __name__ == "__main__":
    main()
