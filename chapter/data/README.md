# Data dictionary

All compositions are mole fractions of Ni, temperatures are in K, and energies
are in J/mol. Phase-field classes are coded 0 = `ok` (the BCC, BCC+FCC, FCC
sequence of the first fitted point between 800 and 2200 K), 1 = `gap` (a
stable BCC+BCC field) and 2 = `other`.

Three ensembles appear with the prefixes `joint_` (2,000 parameter vectors
retained from the two sampling runs), `indep_` (the same values with every
parameter column permuted independently) and `laplace_` (1,000 vectors of the
local Gaussian control). `j400_` is an earlier 400-vector joint ensemble kept
as a regression check.

## `ensembles.npz`

| Array | Shape | Content |
| --- | --- | --- |
| `TT` | 281 | Temperature grid for boundaries, 800 to 2200 K in 5 K steps |
| `MAP_T`, `MAP_X` | 21, 31 | Grid of the decision region: 1200 to 1600 K in 20 K steps, x(Ni) = 0.38 to 0.68 in steps of 0.01 |
| `*_theta` | N × 10 | Parameter vectors, in the order L0 liquid a, b; L0 BCC a, b; L1 BCC a, b; L0 FCC a, b; L1 FCC a, b (a in J/mol, b in J/mol/K) |
| `joint_meta` | 2000 × 3 | Run index, step and walker of each joint vector |
| `*_cls` | N | Phase-field class (0, 1, 2 as above) |
| `*_bcc_bf`, `*_fcc_bf` | N × 281 | BCC-end and FCC-end compositions of the BCC+FCC tie-line on `TT`; NaN where there is none. Where a vector has two BCC+FCC tie-lines at one temperature, these arrays hold the Ni-rich one (see `boundaries_first.npz`) |
| `*_region` | N | Stable assemblage at the decision state, x(Ni) = 0.52 and 1473 K |
| `*_fccmap` | N × 21 × 31 | True where the stable state on the decision grid is single-phase FCC |
| `*_resp` | N × 30 | Predicted response of the 30 candidate measurements, in the order listed in `chapter_numbers.json` (`ranking`): boundary compositions for the first 12, and RT ln a(Cr), referred to pure BCC Cr, for the 18 activity candidates |
| `*_tliq` | N | Lowest temperature at which liquid is stable |

## `boundaries_first.npz`

| Array | Shape | Content |
| --- | --- | --- |
| `*_bcc_bf1`, `*_fcc_bf1` | N × 281 | Compositions of the first (lowest x(Ni)) BCC+FCC tie-line on `TT`, which is the Cr-side boundary discussed in the chapter |
| `*_second_bcc` | N | True if a second, Ni-rich BCC field occurs anywhere within 1200 to 1600 K |

## `boundary_bands.npz`

For each of `joint`, `accepted` (the 220 joint vectors of class 0), `laplace`
and `indep`: `*_bcc_bf` and `*_fcc_bf` (3 × 281) hold the 5th, 50th and 95th
percentiles of the Cr-side boundary compositions on `TT`, and `*_present` (281)
flags the temperatures at which the percentiles are defined.

## `fcc_maps.npz`

`joint`, `indep`, `laplace`, `accepted` and `gap` (21 × 31): fraction of
vectors that predict single-phase FCC on the decision grid, for each ensemble
and for the joint vectors of class 0 and class 1.

## `gap_shapes.npz`

`joint` (2000 × 5), with columns named in `cols`: lowest and highest
temperature with a stable BCC+BCC tie-line, its largest width, and its
smallest left end and largest right end in x(Ni).

## `mode_trace_K100.npz`

`run1_cls`, `run2_cls` (20 × 121): phase-field class of every walker every
100 steps; `run*_steps` gives the steps and `run*_burn` the number of steps
discarded at the start of each run.

## JSON files

| File | Content |
| --- | --- |
| `chapter_numbers.json` | Every number quoted in Section 7: FCC frequencies at the decision state, map statistics, boundary percentiles at 1473 K, measurement errors, and the ranking of the 30 candidates under each assumed error |
| `ensembles_meta.json` | Ensemble sizes, random seeds, and the eigenvalues of the curvature used for the local Gaussian control before and after the three softest were replaced |
| `laplace_summary.json` | Comparison of the local Gaussian control with the joint sample |
| `mode_trace_summary.json` | Class fractions of the walkers before and after the discarded part of each run |
| `setE_mcmc_diag.json` | Autocorrelation times, split R-hat, effective sample sizes and the stopping rule (copy of `results/setE_mcmc_diag.json`) |
| `reference_topology_1200_1550.json` | Phase-field sequence of the Tang–Hallstedt assessment over 1200 to 1550 K |
| `fig_two_modes.json` | Temperatures and energies quoted for the two fitted points |
