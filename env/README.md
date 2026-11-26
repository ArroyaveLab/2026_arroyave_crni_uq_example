# Environments

Two environments were used, because the fit needs analytic gradients that are
not in a released version of ESPEI, whereas the sampling was run on the
released ESPEI 0.9.0.

## Sampling and analysis (stages 4 to 8)

```bash
conda env create -n calphad-uq -f env/calphad-uq.yml
```

`calphad-uq.yml` is the full export of the environment used (Python 3.11,
ESPEI 0.9.0, pycalphad 0.11.2, emcee 2.2.1, arviz 0.23.4).

## Fit and curvature (stages 1 to 3)

```bash
python3.12 -m venv .venv-fit
.venv-fit/bin/pip install -r env/espei-pr268-freeze.txt
git clone https://github.com/dogusariturk/PhaseForgePlus.git
git -C PhaseForgePlus checkout c66656f
export PYTHONPATH="$PWD/PhaseForgePlus/src:$PYTHONPATH"
```

`espei-pr268-freeze.txt` installs ESPEI and pycalphad from the two source
commits listed in the main README. PhaseForge+ was used from a source
checkout, so it is added to the import path and is not in the package list.

## Figures (stage 9)

Any recent Python with numpy, scipy and matplotlib.
