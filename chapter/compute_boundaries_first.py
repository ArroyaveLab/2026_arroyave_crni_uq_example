#!/usr/bin/env python3
"""Cr-side BCC/(BCC+FCC) and (BCC+FCC)/FCC boundaries, and a flag for a second, Ni-rich BCC field.

Why: pdfast.boundaries() keeps the LAST BCC+FCC tie-line it meets at each T. Many gap-class draws have a second,
Ni-rich BCC_A2 field (bcc | bcc+fcc | fcc | fcc+bcc | bcc | bcc+fcc | fcc at 1473 K), and then the returned
"bcc boundary" is the Ni-rich one (x_Ni ~ 0.75). A measurement of the bcc-side boundary on the Cr side reads the
FIRST (lowest-x) BCC+FCC tie-line, so this script stores that one, plus, per draw, whether any T in 1200-1600 K has
more than one BCC+FCC tie-line ("second bcc field") and the region sequence at 1473 K.
usage: compute_boundaries_first.py <nproc> -> data/boundaries_first.npz
"""
import sys
import time
from multiprocessing import get_context

import numpy as np

import common as C

P = C.pdfast
NMIN = 2.5 / (P.NX - 1)
G = {}


def _init():
    G["tt"] = C.ref_grid(C.TT)


def first(th):
    di = P.diagram(th, C.TT, G["tt"])
    xb, xf = np.full(len(C.TT), np.nan), np.full(len(C.TT), np.nan)
    nbf = np.zeros(len(C.TT), np.int8)
    for i, segs in enumerate(di):
        bf = [(xl, xr) for kind, phs, xl, xr in segs
              if kind == 2 and set(phs) == {"BCC_A2", "FCC_A1"} and (xr - xl) >= NMIN]
        nbf[i] = len(bf)
        if bf:
            xb[i], xf[i] = bf[0]
    return xb, xf, nbf


def main():
    nproc = int(sys.argv[1])
    e = np.load(C.DATA / "ensembles.npz")
    out = {}
    t0 = time.time()
    with get_context("fork").Pool(nproc, initializer=_init) as pool:
        for name in ("joint", "indep", "laplace"):
            r = pool.map(first, list(e[f"{name}_theta"]), chunksize=8)
            out[f"{name}_bcc_bf1"] = np.array([a[0] for a in r])
            out[f"{name}_fcc_bf1"] = np.array([a[1] for a in r])
            nbf = np.array([a[2] for a in r])
            m = (C.TT >= 1200) & (C.TT <= 1600)
            out[f"{name}_second_bcc"] = (nbf[:, m] > 1).any(1)
            print(name, round(time.time() - t0), "s; second bcc field fraction", out[f"{name}_second_bcc"].mean(),
                  flush=True)
    np.savez(C.DATA / "boundaries_first.npz", TT=C.TT, **out)


if __name__ == "__main__":
    main()
