"""Reproduce Figures 1 and 4 from the paper."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dualforms.data_generation.supervised import Supervised
from dualforms.problems.heteroscedastic_regression.base import HeteroscedasticRegression
from dualforms.kernels.matern import Matern
from dualforms.plots.heteroscedastic import plot_heteroscedastic

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "main"

# ── Hyperparameters ───────────────────────────────────────────
ALPHA = 0.05
CASE = "case_1"
N = 1000
SEED = 1

MAX_ITERS = 100_000
GAP_TOL = 1e-2
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"
SOLVER = "AGD"

# ── Final hyperparameter choices ────────────────────────────────
PARAMS_JOINT = {
    "mean_theta": 0.4,
    "variance_theta": 0.6,
    "lambda_m": 1e-2,
    "lambda1": 1e-2,
    "lambda2": 1e-2,
}

PARAMS_SEQUENTIAL = {
    "mean_theta": 0.5,
    "variance_theta": 0.6,
    "lambda_m": 1e-1,
    "lambda1": 1e-2,
    "lambda2": 1e-2,
}


def save_prediction_plot(run_dir, label, X, y, X_test, y_test, mean_pred, var_pred, true_mean, true_var, true_lo, true_hi):
    RMSE_mean = np.sqrt(np.mean((mean_pred - true_mean) ** 2))
    RMSE_var = np.sqrt(np.mean((var_pred - true_var) ** 2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_heteroscedastic(
        X, y, X_test, None,
        mean_pred, var_pred,
        true_mean=true_mean, true_lo=true_lo, true_hi=true_hi,
        alpha=ALPHA, title="",
        rmse_mean=RMSE_mean, rmse_var=RMSE_var,
        save_path=run_dir / f"hetreg_{label}_main.pdf",
    )
    plt.close(fig)


def build_model(mode, mean_theta, variance_theta, lambda_m, lambda1, lambda2):
    return HeteroscedasticRegression(
        mode=mode,
        mean_kernel=Matern(),
        variance_kernel=Matern(),
        mean_theta=mean_theta,
        variance_theta=variance_theta,
        lambda_m=lambda_m,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def run(mode, X, y, X_test, params):
    """
    For a fixed training data and feature vectors in test data,
    run a single optimization of the heteroscedastic regression problem at fixed hyperparameters for a given mode,
    and return the mean and variance predictions at each test sample.
    """
    model = build_model(mode, **params)
    model.train(
        X, y,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
        use_low_rank=USE_LOW_RANK,
        target_variance=TARGET_VARIANCE,
        method_low_rank=METHOD_LOW_RANK,
    )

    return model.predict(X_test)


def main():
    # Data
    dataset = Supervised(name=CASE, input_dim=1, output_dim=1)
    X, y = dataset.sample(n=N, seed=SEED)
    X_test, y_test = dataset.sample_grid(n=300, seed=0)
    true_mean, true_var, true_lo, true_hi = dataset.oracle(X_test, alpha=ALPHA)

    # Heteroscedastic regressions
    for mode, params in [("joint", PARAMS_JOINT), ("sequential", PARAMS_SEQUENTIAL)]:
        print(f"\n{'='*70}\n[{mode}]  case={CASE}  solver={SOLVER}  n={N}  \n{'='*70}")
        try:
            mean_pred, var_pred = run(mode, X, y, X_test, params)
            save_prediction_plot(OUT_DIR, mode, X, y, X_test, y_test, mean_pred, var_pred, true_mean, true_var, true_lo, true_hi)
        except Exception as e:
            print(f"FAILED [{mode}]: case={CASE}  solver={SOLVER}  n={N}    ({e})")


if __name__ == "__main__":
    main()
