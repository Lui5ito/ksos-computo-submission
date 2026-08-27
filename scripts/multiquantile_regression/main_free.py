"""Reproduce Figure 2 from the paper."""
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

ZOOM_LEFT = (0.5, 2.5, 0.2, 1.25)
ZOOM_RIGHT = (3.75, 5.75, -1.75, -0.75)

# ── kSoS problem settings ───────────────────────────────────────────────────────────────────
CASE = "case_2"
N = 1000
SEED = 5

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

MAX_ITERS = 100_000
GAP_TOL = 1e-2
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"
SOLVER = "AGD"

# ── Best hyperparameter combination identified by cross-validation (see appendix_free.py) ───
PARAMS_FREE = {
    "theta_m": 0.7,
    "theta_qlow": 1.3,
    "theta_qupp": 0.7,
    "lambda_m": 1,
    "lambda1": 1,
    "lambda2": 1,
}
PARAMS_SAME = {
    "theta_m": 0.7,
    "theta_qlow": 0.7,
    "theta_qupp": 0.7,
    "lambda_m": 1,
    "lambda1": 1,
    "lambda2": 1,
}

def build_model(theta_m, theta_qlow, theta_qupp, lambda_m, lambda1, lambda2):
    n_q = len(QUANTILE_LEVELS)
    return MultiQuantileRegression(
        mode="sequential",
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


def run_sequential(X, y, X_test, params):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run a single optimization for the kernel quantile regression problem for a fixed set of hyperparameters, then,
    run a single optimization for the multi-quantile regression with known median problem for a fixed set of hyperparameters,
    and return the quantile predictions at each test samples.
    """
    model = build_model(**params)
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
    for label, params in [("same", PARAMS_SAME), ("free", PARAMS_FREE)]:
        print(f"\n{'='*70}\n[{label}]  case={CASE}  solver={SOLVER}  n={N}\n{'='*70}")
        try:
            predictions[label] = run_sequential(X, y, X_test, params)
        except Exception as e:
            print(f"FAILED [{label}]: case={CASE}  solver={SOLVER}  n={N}  ({e})")

    # Final plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "multiquant_same_vs_free_main.pdf"
    fig = plot_multiquantile_comparison(
        X, y, X_test, y_test,
        predictions_by_row=[predictions["same"], predictions["free"]],
        true_quantiles=true_quantiles,
        quantile_levels=QUANTILE_LEVELS,
        row_labels=[r"Same $\theta^f$", r"Free $\theta^f$"],
        rmse_by_row=[
            aggregate_rmse(predictions["same"], true_quantiles),
            aggregate_rmse(predictions["free"], true_quantiles),
        ],
        zoom_left=ZOOM_LEFT, zoom_right=ZOOM_RIGHT,
        save_path=save_path,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
