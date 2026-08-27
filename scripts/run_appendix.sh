#!/usr/bin/env bash
source .venv/bin/activate

python -u experiments/multiquantile_regression/appendix_joint.py --n_cores 100
python -u experiments/multiquantile_regression/appendix_free.py --n_cores 100

python -u experiments/heteroscedastic_regression/appendix.py --n_cores 100

python -u experiments/density_estimation/appendix.py --n_cores 100