#!/usr/bin/env python3
"""Shape of the stable BCC_A2+BCC_A2 gap in every joint draw (1 K steps would be costly; TT grid, 5 K):
lowest/highest T with a stable bcc+bcc tie-line, its widest extent, and its largest Ni-side end.
usage: compute_gap_shapes.py <nproc> -> data/gap_shapes.npz
"""
import sys
from multiprocessing import get_context

import numpy as np

import common as C

P = C.pdfast
NMIN = 2.5 / (P.NX - 1)
G = {}


def _init():
    G["tt"] = C.ref_grid(C.TT)


def shape(th):
    di = P.diagram(th, C.TT, G["tt"])
    rows = []
    for t, segs in zip(C.TT, di):
        for kind, phs, xl, xr in segs:
            if kind == 2 and phs[0] == phs[1] == "BCC_A2" and xr - xl > NMIN:
                rows.append((t, xl, xr))
    if not rows:
        return [np.nan] * 5
    a = np.array(rows)
    return [a[:, 0].min(), a[:, 0].max(), (a[:, 2] - a[:, 1]).max(), a[:, 1].min(), a[:, 2].max()]


def main():
    e = np.load(C.DATA / "ensembles.npz")
    with get_context("fork").Pool(int(sys.argv[1]), initializer=_init) as pool:
        r = np.array(pool.map(shape, list(e["joint_theta"]), chunksize=8))
    np.savez(C.DATA / "gap_shapes.npz", cols=np.array(["T_lo", "T_hi", "max_width", "x_left_min", "x_right_max"]),
             joint=r)
    print("done", np.isfinite(r[:, 0]).mean())


if __name__ == "__main__":
    main()
