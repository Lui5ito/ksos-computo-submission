#!/usr/bin/env bash
source .venv/bin/activate

#python -u experiments/multiquantile_regression/main_free.py
#python -u experiments/multiquantile_regression/main_joint.py

python -u dual-formulations/experiments/density_estimation/main.py
#python -u experiments/heteroscedastic_regression/main.py