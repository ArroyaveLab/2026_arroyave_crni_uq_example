# Cr–Ni uncertainty-analysis example

Data and plotting scripts for the computational example (Section 7) of

> R. Arróyave, "Beyond Quantification: Propagating and Exploiting CALPHAD
> Uncertainty in Materials Design," in *Uncertainty Quantification in
> Multiscale Materials Modeling*, 2nd ed., D. L. McDowell and Y. Wang (eds.),
> Elsevier (in preparation).

The example fits and samples a thermodynamic description of the Cr–Ni system
(liquid, BCC_A2 and FCC_A1 solution phases, ten fitted coefficients), compares
the predicted stability of single-phase FCC under joint sampling, independent
recombination of the parameters, and a local Gaussian approximation, and ranks
30 candidate measurements by the reduction that they are expected to produce
in the uncertainty of the FCC classification.

## Layout

The repository mirrors the directory layout of the original run, so that the
scripts find their inputs through relative paths.

| Path | Content |
| --- | --- |
| `inputs/` | Model definition (`cr-ni-model.tdb`, `CR-NI_phases.json`), the upstream ESPEI datasets (`chapter-input-data/`, `hao-thermo/`, `recipe-zpf/`), the derived dataset collections A to E (`derived/`; the chapter uses `setE_orchid_sigmas`), the curvature at the two fitted points (`setE_spark/`), and the independent Tang–Hallstedt assessment (`reference/`, CC BY 4.0, unchanged) |
| `scripts/` | The pipeline: `build_inputs.py`, `build_setD.py`, `build_setE.py`, `fit.py`, `curvature.py`, `mcmc.py`, `analyze_mcmc.py`, with `crni.py` (model, prior, posterior) and `pdfast.py` (phase-diagram calculation) |
| `fit/`, `results/`, `logs/` | Fitted points, summaries and logs of the fits |
| `mcmc/setE_run1`, `mcmc/setE_run2` | The two ensemble-sampling runs (20 walkers, 12,000 steps each) |
| `chapter/` | Analysis for Section 7: `compute_ensembles.py`, `compute_boundaries_first.py`, `compute_gap_shapes.py`, `compute_mode_trace.py`, `analyze_decisions.py`, their output in `chapter/data/`, and figures in `chapter/figures/` |
| `plots/` | Scripts that draw the figures printed in the chapter from `chapter/data/` |
| `tests/` | Consistency checks used during the fit |
| `env/` | The two software environments |

The arrays and JSON files in `chapter/data/` are described in
[`chapter/data/README.md`](chapter/data/README.md).

**Material that precedes set E.** The chapter uses only the dataset
collection `setE_orchid_sigmas`. Collections A to D are the steps that led to
it and are kept because set E is built from set D: A and B apply the data
treatment of the published Cr–Ni assessment with and without one early
dataset, C assigns weights from stated uncertainties, D keeps every vertex of
the phase-boundary records and adds the penalty on excluded phases, and E
replaces the uncertainties of the thermochemical records that carry none by
the ORCHID rule. `fit/`, `results/` and `logs/` therefore also contain the
fits of sets A to D. `results/notconverged/` holds an earlier 400-vector
ensemble that the analysis rebuilds as a regression check. The scripts in
`tests/` are one-off checks made during the fit; each states its purpose in
its first lines and is run from the repository root.

## Stages and commands

Run from the repository root.

| Stage | Command | Environment | Approximate cost |
| --- | --- | --- | --- |
| 1. Build dataset collections | `python scripts/build_inputs.py`, `build_setD.py`, `build_setE.py` | fit | seconds |
| 2. Fit (multistart) | `python scripts/fit.py setE_orchid_sigmas <nproc>` | fit | about 5 min per start |
| 3. Curvature at a fitted point | `python scripts/curvature.py setE_orchid_sigmas [theta.json]` | fit | minutes |
| 3b. Soft directions of the curvature | `python scripts/soft_scan.py scripts setE_orchid_sigmas <curvature.json> > <scan.json>`, then `python scripts/repair_cov.py <curvature.json> <scan.json> <out.json>` | fit | minutes |
| 4. Sampling | `python scripts/mcmc.py setE_orchid_sigmas mcmc/setE_run1 <seed> 20 12000 <nproc>` | sampling | about two days per run on 6 cores |
| 5. Sampling diagnostics | `python scripts/analyze_mcmc.py mcmc/setE_run1 mcmc/setE_run2 results/setE_mcmc_diag.json` | sampling | minutes |
| 6. Ensembles | `python chapter/compute_ensembles.py 2000 1000 <nproc>` | sampling | about 22 min on 3 cores |
| 7. Boundaries, gap shapes, walker classes | `python chapter/compute_boundaries_first.py <nproc>`, `compute_gap_shapes.py <nproc>`, `compute_mode_trace.py 100 <nproc>` | sampling | minutes |
| 8. Decisions and measurement ranking | `python chapter/analyze_decisions.py` | sampling | seconds |
| 9. Figures | `for f in plots/*.py; do python "$f"; done` | numpy, scipy, matplotlib | seconds |

Random seeds: 1105101 and 2211202 for the two sampling runs, 20261106 for the
independent recombination and 20261107 for the local Gaussian control.

Each sampling run starts from walkers drawn around one of the two fitted
points, with the curvature at that point. Before stage 4, copy the
corresponding file to the name that `mcmc.py` reads:

```bash
cp inputs/setE_spark/curvature_setE_run1_b557.json fit/curvature_setE_orchid_sigmas.json   # first run
cp inputs/setE_spark/curvature_setE_run2_b069.json fit/curvature_setE_orchid_sigmas.json   # second run
```

The figure of the two fitted points is drawn by `chapter/fig_two_modes.py`
(run from `chapter/`), and `plots/build_fig_27_context_overview.py` crops its
first panel for the context figure (requires `pdfcrop`, `pypdf` and
`reportlab`).

## Environments

Installation commands for both environments are in
[`env/README.md`](env/README.md).

- **Sampling and analysis** (`env/calphad-uq.yml`): Python 3.11, ESPEI 0.9.0,
  pycalphad 0.11.2, emcee 2.2.1, numpy 2.4.6, scipy 1.17.1, arviz 0.23.4.
- **Fit and curvature** (`env/espei-pr268-freeze.txt`): Python 3.12, ESPEI with
  analytic gradients (`github.com/cjkunselman18/ESPEI`, commit `9039ea2`),
  pycalphad (`github.com/pycalphad/pycalphad`, commit `9a6939d`), and the hinge
  penalty of PhaseForge+ (`github.com/dogusariturk/PhaseForgePlus`). The hinge
  module was last changed before the fit in commit `965f8ad` (2026-10-31), and
  commit `c66656f` (2026-10-31) is the snapshot pinned in later reruns.

## What has been verified from this repository

Every stage was rerun from a fresh copy of this repository on an Apple M1 Max
and compared with the archived results.

| Stage | Result |
| --- | --- |
| 1. Dataset collections A to E | Every dataset file identical |
| 2. Fit (eight starts) | Identical log posterior and parameter values at every start, including the two fitted points (-6153.70 and -6149.93) |
| 3, 3b. Curvature at both fitted points, with the three softest directions replaced | Identical eigenvalues; covariance equal to the stored one to a relative difference of 5e-9 |
| 4. Sampling | Not reproduced step by step, and not expected to be (see below) |
| 5. Sampling diagnostics | Identical |
| 6. Ensembles | `ensembles.npz` identical, all 35 arrays |
| 7. Boundaries, gap shapes, walker classes | Identical |
| 8. Decisions and measurement ranking | `chapter_numbers.json`, `fcc_maps.npz` and `boundary_bands.npz` identical |
| 9. Figures | All scripts in `plots/` and `chapter/fig_two_modes.py` run and regenerate the figures |

**Sampling.** The two stored runs in `mcmc/` are the record of this analysis,
and everything downstream is computed from them. Ensemble sampling is a
stochastic process whose trajectory amplifies differences in the last digits
of the arithmetic. A run repeated with the same seed on a different computer
starts from the same walkers to within 1e-12 and follows a different
trajectory after a few steps, so that it is a new realization of the same
process and should be compared with the stored runs statistically. The full
12,000-step runs were not repeated.

## Known limits of the record

- The exact PhaseForge+ commit loaded at the time of the fit was not logged
  (see Environments). At the time of writing, neither the ESPEI fork nor
  PhaseForge+ states a license.
- Stages 2 and 3 evaluate a log posterior that includes the hinge penalty on
  excluded phases (290 cells, `inputs/derived/setE_orchid_sigmas/hinge/`),
  whereas stage 4 samples the ESPEI 0.9.0 log posterior without it. The
  penalty contributes -0.06 and -0.34 to the log posterior at the two fitted
  points.
- The cached reference-energy grids (`*_ref_*.npy`) are not stored; they are
  regenerated on first use.
- The figure scripts in `chapter/` use `chapter/style.py`, which looks for
  Arial in the macOS system fonts and falls back to the default font
  elsewhere.

**Third-party data.** `inputs/chapter-input-data/` comes from
`github.com/PhasesResearchLab/ESPEI-datasets` (CC BY 4.0, checksums in
`inputs/UPSTREAM.json`). The liquid activity and mixing-enthalpy datasets in
`inputs/hao-thermo/` were digitized from figures in Hao, Ruban and Xiong,
*Calphad* 73 (2021) 102268; each file cites the original measurement.

## Limits

The slowest parameter directions of the two sampling runs did not meet the
stopping rule recorded in `data/setE_mcmc_diag.json`, the model contains only
three solution phases, and stable phases are determined on a finite
composition grid. The frequencies should therefore not be read as precise
posterior probabilities.

## License

Code in `scripts/` is released under the MIT License (`LICENSE`). Data in
`data/` and the figures are released under the Creative Commons Attribution
4.0 International License (`LICENSE-DATA`).
