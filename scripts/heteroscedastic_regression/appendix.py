"""Reproduce Tables 1 and 5 from Appendix B of the paper."""
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
from dualforms.problems.mean_regression.dual import Dual_Mean
from dualforms.problems.heteroscedastic_regression.dual import Dual_Heteroscedastic
from dualforms.problems.heteroscedastic_regression.dual_known_mean import Dual_Heteroscedastic_KnownMean
from dualforms.kernels.matern import Matern
from dualforms.cross_validation.grid_search import GridSearch
from dualforms.plots.heteroscedastic import plot_heteroscedastic

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "appendix"

# ── kSoS problem settings ────────────────────────────────────────
ALPHA = 0.05
CASE = "case_1"
N = 1000
SEED = 1

N_FOLDS = 5
MAX_ITERS = 100_000
GAP_TOL = 1e-2
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"
SOLVER = "AGD"

# ── Hyperarameter grids for cross-validation ─────────────────────
THETA_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_GRID = [1e-2, 1e-1, 1]

PARAM_GRID_JOINT = {
    "mean_theta": THETA_GRID,
    "variance_theta": THETA_GRID,
    "lambda_m": LAMBDA_GRID,
    "lambda1": LAMBDA_GRID,
    "lambda2": LAMBDA_GRID,
}
PARAM_GRID_MEAN = {
    "theta": THETA_GRID,
    "lambda_m": LAMBDA_GRID,
}
PARAM_GRID_VAR = {
    "theta": THETA_GRID,
    "lambda1": LAMBDA_GRID,
    "lambda2": LAMBDA_GRID,
}


def build_model_joint(mean_theta, variance_theta, lambda_m, lambda1, lambda2):
    return Dual_Heteroscedastic(
        mean_kernel=Matern(),
        variance_kernel=Matern(),
        mean_theta=mean_theta,
        variance_theta=variance_theta,
        lambda_m=lambda_m,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def build_model_mean(theta, lambda_m):
    return Dual_Mean(
        mean_kernel=Matern(),
        mean_theta=theta,
        lambda_m=lambda_m,
    )


def build_model_var(theta, lambda1, lambda2):
    return Dual_Heteroscedastic_KnownMean(
        variance_kernel=Matern(),
        variance_theta=theta,
        lambda1=lambda1,
        lambda2=lambda2,
    )


def save_prediction_plot(run_dir, label, X, y, X_test, y_test, mean_pred, var_pred, true_mean, true_var, true_lo, true_hi):
    RMSE_mean = np.sqrt(np.mean((mean_pred - true_mean) ** 2))
    RMSE_var = np.sqrt(np.mean((var_pred - true_var) ** 2))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = plot_heteroscedastic(
        X, y, X_test, y_test,
        mean_pred, var_pred,
        true_mean=true_mean, true_lo=true_lo, true_hi=true_hi,
        alpha=ALPHA, title="",
        rmse_mean=RMSE_mean, rmse_var=RMSE_var,
        save_path=run_dir / f"hetreg_{label}.pdf",
    )
    plt.close(fig)


def run_joint(X, y, X_test, n_cores, run_dir):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run one cross-validation estimation of the hyperparameters for the joint heteroscedastic regression problem,
    generate a LaTeX table of the cross-validation results,
    and return the mean and variance predictions at each test sample with best hyperparameter combination.
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

    gs.to_latex_table(
        save_path=run_dir / "hetreg_cv_results_joint.tex",
        caption="CV results for the joint heteroscedastic regression.",
        label="tab:cv_heteroscedastic_joint",
        top_k=None,
        param_labels={
            "mean_theta": r"\(\theta^{g}\)",
            "variance_theta": r"\(\theta^{f}\)",
            "lambda_m": r"\(\lambda^{g}\)",
            "lambda1": r"\(\lambda_{1}^{f}\)",
            "lambda2": r"\(\lambda_{2}^{f}\)",
        },
    )

    return gs.best_model.predict(X_test)


def run_sequential(X, y, X_test, n_cores, run_dir):
    """
    For a fixed training data (X,y) and feature vectors in test data X_test,
    run one cross-validation estimation of the hyperparameters for the kernel ridge regression problem, then 
    run one cross-validation estimation of the hyperparameters for the heteroscedastic regression with known mean problem 
    (with mean given by the best combination from the first cross-validation), generate a LaTeX table of the cross-validation results,
    and return the mean and variance predictions at each test sample.
    """
    gs_mean = GridSearch(build_model=build_model_mean, param_grid=PARAM_GRID_MEAN, n_folds=N_FOLDS, n_jobs=n_cores)
    gs_mean.train(
        X=X,
        y=y,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
    )
    predictions_mean_train = gs_mean.best_model.mean
    print(f"[mean] best params: {gs_mean.best_params}")

    gs_mean.to_latex_table(
        save_path=run_dir / "hetreg_cv_results_sequential_mean.tex",
        caption="CV results for the mean regression.",
        label="tab:cv_heteroscedastic_sequential_mean",
        top_k=None,
        param_labels={
            "theta": r"\(\theta^{\mu}\)",
            "lambda_m": r"\(\lambda^{\mu}\)",
        },
    )

    gs = GridSearch(build_model=build_model_var, param_grid=PARAM_GRID_VAR, n_folds=N_FOLDS, n_jobs=n_cores)
    gs.train(
        X=X,
        y=y,
        aux=predictions_mean_train,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        gap_tol=GAP_TOL,
        use_low_rank=USE_LOW_RANK,
        target_variance=TARGET_VARIANCE,
        method_low_rank=METHOD_LOW_RANK,
    )
    print(f"[variance] best params: {gs.best_params}")

    gs.to_latex_table(
        save_path=run_dir / "hetreg_cv_results_sequential_variance.tex",
        caption="CV results for the variance regression.",
        label="tab:cv_heteroscedastic_sequential_variance",
        top_k=None,
        param_labels={
            "variance_theta": r"\(\theta^{f}\)",
            "lambda1": r"\(\lambda_{1}^{f}\)",
            "lambda2": r"\(\lambda_{2}^{f}\)",
        },
    )

    return gs_mean.best_model.predict(X_test), gs.best_model.predict(X_test)


def main(n_cores=1):
    # Sample data
    dataset  = Supervised(name=CASE, input_dim=1, output_dim=1)
    X, y     = dataset.sample(n=N, seed=SEED)
    X_test, y_test = dataset.sample_grid(n=300, seed=0)
    true_mean, true_var, true_lo, true_hi = dataset.oracle(X_test, alpha=ALPHA)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Run both heteroscedastic regression variants
    for label, run in [("joint", run_joint), ("sequential", run_sequential)]:
        print(f"\n{'='*70}\n[{label}]  case={CASE}  solver={SOLVER}  n={N}  \n{'='*70}")
        try:
            mean_pred, var_pred = run(X, y, X_test, n_cores, OUT_DIR)
            save_prediction_plot(OUT_DIR, label, X, y, X_test, y_test, mean_pred, var_pred, true_mean, true_var, true_lo, true_hi)
        except Exception as e:
            print(f"FAILED [{label}]: case={CASE}  solver={SOLVER}  n={N}    ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=1)
    args = parser.parse_args()
    main(n_cores=args.n_cores)
