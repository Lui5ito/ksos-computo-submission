"""Reproduce Tables 6 and 7 from Appendix B of the paper."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dualforms.data_generation.supervised import Supervised
from dualforms.problems.median_regression.dual import Dual_Median
from dualforms.problems.multiquantile_regression.dual import Dual_MultiQuantile
from dualforms.problems.multiquantile_regression.dual_known_median import Dual_MultiQuantile_KnownMedian
from dualforms.kernels.matern import Matern
from dualforms.cross_validation.grid_search import GridSearch
from dualforms.plots.multiquantile import compute_true_quantiles, plot_multiquantile_comparison

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "appendix"

ZOOM_LEFT = (0.1, 2.1, -0.5, 2)
ZOOM_RIGHT = (3.8, 5.8, -1.9, -0.2)

# ── kSoS problem settings ────────────────────────────────────────
CASE = "case_2"
N = 1000
SEED = 3

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]

N_FOLDS = 5
MAX_ITERS = 100_000
GAP_TOL = 1e-2
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"
SOLVER = "AGD"

# ── Hyperarameter grids for cross-validation ─────────────────────
THETA_GRID = [0.7, 0.9, 1.1, 1.3]
LAMBDA_GRID = [1]

PARAM_GRID_MEDIAN = {
    "theta": THETA_GRID,
    "lambda_m": LAMBDA_GRID,
}
PARAM_GRID_SEQUENTIAL = {
    "theta_qlow": THETA_GRID,
    "theta_qupp": THETA_GRID,
    "lambda1": LAMBDA_GRID,
    "lambda2": LAMBDA_GRID,
}
PARAM_GRID_JOINT = {
    "theta_m": THETA_GRID,
    "theta_qlow": THETA_GRID,
    "theta_qupp": THETA_GRID,
    "lambda_m": LAMBDA_GRID,
    "lambda1": LAMBDA_GRID,
    "lambda2": LAMBDA_GRID,
}


def build_model_median(theta, lambda_m):
    return Dual_Median(
        median_kernel=Matern(),
        median_theta=theta,
        lambda_m=lambda_m,
    )


def build_model_sequential(theta_qlow, theta_qupp, lambda1, lambda2):
    n_q = len(QUANTILE_LEVELS)
    return Dual_MultiQuantile_KnownMedian(
        quantile_levels=QUANTILE_LEVELS,
        quantile_kernels=[Matern() for _ in range(n_q)],
        quantile_thetas=[theta_qlow] * (n_q // 2) + [theta_qupp] * (n_q // 2),
        lambda1s=[lambda1] * n_q,
        lambda2s=[lambda2] * n_q,
    )


def build_model_joint(theta_m, theta_qlow, theta_qupp, lambda_m, lambda1, lambda2):
    n_q = len(QUANTILE_LEVELS)
    return Dual_MultiQuantile(
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


def run_sequential(X, y, X_test, n_cores):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run one cross-validation estimation of the hyperparameters for the kernel quantile regression problem, then
    run one cross-validation estimation of the hyperparameters for the multi-quantile regression with known median problem 
    (with medianan given by the best combination from the first cross-validation) and 
    with different lengthscales for the quantiles under and above the median,
    generate LaTeX tables of the cross-validation results,
    and finally return the quantile predictions at each test samples with best hyperparameter combination.
    """
    gs_median = GridSearch(build_model=build_model_median, param_grid=PARAM_GRID_MEDIAN, n_folds=N_FOLDS, n_jobs=n_cores)
    gs_median.train(
        X=X,
        y=y,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
    )
    predictions_median_train = gs_median.best_model.median
    predictions_median_test  = gs_median.best_model.predict(X_test)
    print(f"[median] best params: {gs_median.best_params}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gs_median.to_latex_table(
        save_path=OUT_DIR / "multiquant_joint_cv_results_median.tex",
        caption="CV results for the median.",
        label="tab:cv_multiquantile_sequential_median",
        top_k=None,
        param_labels={
            "theta": r"\(\theta^{q_{0.5}}\)",
        },
        exclude_params=["lambda_m"],
    )

    gs = GridSearch(build_model=build_model_sequential, param_grid=PARAM_GRID_SEQUENTIAL, n_folds=N_FOLDS, n_jobs=n_cores)
    gs.train(
        X=X,
        y=y,
        aux=predictions_median_train,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
        use_low_rank=USE_LOW_RANK,
        target_variance=TARGET_VARIANCE,
        method_low_rank=METHOD_LOW_RANK,
    )
    print(f"[quantiles] best params: {gs.best_params}")

    gs.to_latex_table(
        save_path=OUT_DIR / "multiquant_joint_cv_results_quantiles.tex",
        caption="CV results for the quantiles.",
        label="tab:cv_multiquantile_sequential_quantiles",
        top_k=None,
        param_labels={
            "theta_qlow": r"\(\theta_{low}^{f}\)",
            "theta_qupp": r"\(\theta_{upp}^{f}\)",
        },
        exclude_params=["lambda1", "lambda2"],
    )

    return gs.best_model.predict(X_test, predictions_median_test)


def run_joint(X, y, X_test, n_cores):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run one cross-validation estimation of the hyperparameters for the joint multi-quantile regression problem and 
    with different lengthscales for the quantiles under and above the median,
    generate a LaTeX table of the cross-validation results,
    and finally return the quantile predictions at each test samples with best hyperparameter combination.
    """
    gs = GridSearch(build_model=build_model_joint, param_grid=PARAM_GRID_JOINT, n_folds=N_FOLDS, n_jobs=n_cores)
    gs.train(
        X=X,
        y=y,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
        use_low_rank=USE_LOW_RANK,
        target_variance=TARGET_VARIANCE,
        method_low_rank=METHOD_LOW_RANK,
    )
    print(f"[joint] best params: {gs.best_params}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gs.to_latex_table(
        save_path=OUT_DIR / "multiquant_joint_cv_results_joint.tex",
        caption="CV results for the joint multi-quantile regression.",
        label="tab:cv_multiquantile_joint",
        top_k=None,
        param_labels={
            "theta_m": r"\(\theta^{g}\)",
            "theta_qlow": r"\(\theta_{low}^{f}\)",
            "theta_qupp": r"\(\theta_{upp}^{f}\)",
        },
        exclude_params=["lambda_m", "lambda1", "lambda2"],
    )

    return gs.best_model.predict(X_test)


def main(n_cores=1):
    # Sample data
    dataset = Supervised(name=CASE, input_dim=1, output_dim=1)
    X, y = dataset.sample(n=N, seed=SEED)
    X_test, y_test = dataset.sample_grid(n=300, seed=0)
    true_quantiles = compute_true_quantiles(dataset, X_test, QUANTILE_LEVELS)

    # Cross-validation for both quantile regressions variants
    predictions = {}
    for label, run in [("sequential", run_sequential), ("joint", run_joint)]:
        print(f"\n{'='*70}\n[{label}]  case={CASE}  solver={SOLVER}  n={N}  \n{'='*70}")
        try:
            predictions[label] = run(X, y, X_test, n_cores)
        except Exception as e:
            print(f"FAILED [{label}]: case={CASE}  solver={SOLVER}  n={N}    ({e})")

    # Final plot
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUT_DIR / "multiquant_joint.pdf"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=1)
    args = parser.parse_args()
    main(n_cores=args.n_cores)
