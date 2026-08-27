"""Reproduce Figure 5 from the paper."""
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
from dualforms.problems.multiquantile_regression.base import MultiQuantileRegression
from dualforms.kernels.matern import Matern
from dualforms.plots.multiquantile import compute_true_quantiles, plot_multiquantile_comparison

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "main"

ZOOM_LEFT = (0.1, 2.1, -0.5, 2)
ZOOM_RIGHT = (3.8, 5.8, -1.9, -0.2)

# ── kSoS problem settings ────────────────────────────────────────────────────────────────────
CASE = "case_2"
N = 1000
SEED = 3

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

MAX_ITERS = 100_000
GAP_TOL = 1e-2
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"
SOLVER = "AGD"

# ── Best hyperparameter combination identified by cross-validation (see appendix_joint.py) ───
PARAMS_SEQUENTIAL = {
    "theta_m": 0.7,
    "theta_qlow": 0.9,
    "theta_qupp": 0.7,
    "lambda_m": 1,
    "lambda1": 1,
    "lambda2": 1,
}
PARAMS_JOINT = {
    "theta_m": 0.9,
    "theta_qlow": 1.3,
    "theta_qupp": 0.7,
    "lambda_m": 1,
    "lambda1": 1,
    "lambda2": 1,
}


def build_model(mode, theta_m, theta_qlow, theta_qupp, lambda_m, lambda1, lambda2):
    n_q = len(QUANTILE_LEVELS)
    return MultiQuantileRegression(
        mode=mode,
        quantile_levels=QUANTILE_LEVELS,
        median_kernel=Matern(),
        quantile_kernels=[Matern() for _ in range(n_q)],
        median_theta=theta_m,
        quantile_thetas=[theta_qlow] * (n_q // 2) + [theta_qupp] * (n_q // 2),
        lambda_m=lambda_m,
        lambda1s=[lambda1] * n_q,
        lambda2s=[lambda2] * n_q,
    )


def aggregate_rmse(predictions, true_quantiles):
    return np.mean([
        np.sqrt(np.mean((predictions[tau] - true_quantiles[tau]) ** 2))
        for tau in predictions.keys()
    ])

def median_rmse(predictions, true_quantiles):
    return np.sqrt(np.mean((predictions[0.5] - true_quantiles[0.5]) ** 2))


def run(mode, X, y, X_test, params):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run a single optimization for the multi-quantile regression problem for a fixed set of hyperparameters 
    (sequential or joint depending on mode),
    and return the quantile predictions at each test samples.
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
    # Sample data
    dataset = Supervised(name=CASE, input_dim=1, output_dim=1)
    X, y = dataset.sample(n=N, seed=SEED)
    X_test, y_test = dataset.sample_grid(n=300, seed=0)
    true_quantiles = compute_true_quantiles(dataset, X_test, QUANTILE_LEVELS)

    # Run both quantile regression variants
    predictions = {}
    for mode, params in [("sequential", PARAMS_SEQUENTIAL), ("joint", PARAMS_JOINT)]:
        print(f"\n{'='*70}\n[{mode}]  case={CASE}  solver={SOLVER}  n={N}\n{'='*70}")
        try:
            predictions[mode] = run(mode, X, y, X_test, params)
        except Exception as e:
            print(f"FAILED [{mode}]: case={CASE}  solver={SOLVER}  n={N} ({e})")

    # Final plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "multiquant_sequential_vs_joint_main.pdf"
    fig = plot_multiquantile_comparison(
        X, y, X_test, y_test,
        predictions_by_row=[predictions["sequential"], predictions["joint"]],
        true_quantiles=true_quantiles,
        quantile_levels=QUANTILE_LEVELS,
        row_labels=["Sequential", "Joint"],
        rmse_by_row=[
            aggregate_rmse(predictions["sequential"], true_quantiles),
            aggregate_rmse(predictions["joint"], true_quantiles),
        ],
        rmse_median_by_row=[
            median_rmse(predictions["sequential"], true_quantiles),
            median_rmse(predictions["joint"], true_quantiles),
        ],
        zoom_left=ZOOM_LEFT, zoom_right=ZOOM_RIGHT,
        save_path=save_path,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
