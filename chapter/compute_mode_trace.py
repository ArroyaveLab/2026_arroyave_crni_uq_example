#!/usr/bin/env python3
"""Per-walker topology class (ok / gap / other) along both set E chains, every K steps from step 0.
Same classification as cr-ni-setH-20261106/scripts/mode_occupancy.py. Run niced on spare cores.
usage: compute_mode_trace.py <K> <nproc>   ->  data/mode_trace_K<K>.npz
Cost: one pdfast.topology call per (step, walker), ~0.34 s on the loaded Studio.
"""
import sys
import time
from multiprocessing import get_context

import numpy as np

import common as C

REF = None


def _init():
    global REF
    REF = C.ref_grid(C.TT)


def _cls(th):
    return C.CLASSES.index(C.classify(th, REF)[0])


def main():
    K, nproc = int(sys.argv[1]), int(sys.argv[2])
    out = {}
    t0 = time.time()
    for k, d in enumerate(C.RUNS):
        ch = np.load(C.RUN / d / "chain.npy")
        steps = np.arange(0, ch.shape[0], K)
        if steps[-1] != ch.shape[0] - 1:
            steps = np.append(steps, ch.shape[0] - 1)
        with get_context("fork").Pool(nproc, initializer=_init) as pool:
            lab = pool.map(_cls, [ch[s, w] for s in steps for w in range(ch.shape[1])], chunksize=10)
        out[f"run{k + 1}_steps"] = steps
        out[f"run{k + 1}_cls"] = np.array(lab, np.int8).reshape(len(steps), ch.shape[1]).T  # (walkers, steps)
        out[f"run{k + 1}_burn"] = C.BURN[d]
        print(d, len(steps), "steps", round(time.time() - t0), "s", flush=True)
    np.savez(C.DATA / f"mode_trace_K{K}.npz", K=K, classes=np.array(C.CLASSES), **out)


if __name__ == "__main__":
    main()
