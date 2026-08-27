"""Reproduce Table 4 from Appendix B of the paper."""
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

from dualforms.problems.density_estimation.dual import Dual_Density
from dualforms.kernels.matern import Matern
from dualforms.cross_validation.grid_search import GridSearch
from dualforms.plots.density2D import plot_density_2D
from dualforms.data_generation.unsupervised import Unsupervised

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "appendix"

# ── kSoS problem settings ────────────────────────────────────────
N = 2000
SEED = 7
SOLVER = "BFGS"

N_FOLDS = 5
MAX_ITERS = 100_000
COV_TOL = 1e-2
GAP_TOL = 1e-2
REF_DENSITY = "gaussian"
N_MC_DENSITY = 10_000
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"

N_INTEGRAL = 1000
SEED_INTEGRAL = 10

# ── Hyperarameter grids for cross-validation ─────────────────────
THETA_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
LAMBDA_GRID = [1e-2, 1e-1, 1]

PARAM_GRID = {
    "theta_x1": THETA_GRID,
    "theta_x2": THETA_GRID,
    "lambda1": LAMBDA_GRID,
    "lambda2": LAMBDA_GRID,
}

def build_model(theta_x1, theta_x2, lambda1, lambda2):
    return Dual_Density(kernel=Matern(), theta=[theta_x1, theta_x2], lambda1=lambda1, lambda2=lambda2)


def run_one(n, seed, n_cores):
    """
    For a fixed training size and seed, generate the training sample,
    run one cross-validation estimation of the hyperparameters, 
    generate a LaTeX table of the cross-validation results,
    evaluate the true density and plot the true density versus estimation with best hyperparameter combination.
    """
    run_dir = OUT_DIR
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = Unsupervised(name="banana")
    X_train = dataset.sample(n, seed)

    gs = GridSearch(build_model=build_model, param_grid=PARAM_GRID, n_folds=N_FOLDS, n_jobs=n_cores)

    gs.train(
        X=X_train,
        y=None,
        ref_density=REF_DENSITY,
        n_mc_density=N_MC_DENSITY,
        solver=SOLVER,
        max_iters=MAX_ITERS,
        cov_tol=COV_TOL,
        gap_tol=GAP_TOL,
        use_low_rank=USE_LOW_RANK,
        target_variance=TARGET_VARIANCE,
        method_low_rank=METHOD_LOW_RANK,
    )

    print(f"[density] best params: {gs.best_params}")

    gs.to_latex_table(
        save_path=run_dir / "density_cv_results.tex",
        caption="CV results for density estimation.",
        label="tab:cv_density_estimation",
        top_k=None,
        param_labels={
            "theta_x1": r"\(\theta_{x_1}^{f}\)",
            "theta_x2": r"\(\theta_{x_2}^{f}\)",
            "lambda1": r"\(\lambda_{1}^{f}\)",
            "lambda2": r"\(\lambda_{2}^{f}\)",
        },
    )

    sample_min = X_train.min(axis=0)
    sample_max = X_train.max(axis=0)
    x_grid, y_grid, dx, dy = dataset.sample_grid(n=200, sample_min=sample_min, sample_max=sample_max)

    density_pred = gs.best_model.predict(np.vstack([x_grid.ravel(), y_grid.ravel()]).T).reshape(x_grid.shape)
    density_pred = density_pred / (np.sum(density_pred) * dx * dy)

    density_true = dataset.oracle(x_grid, y_grid, dx, dy)

    integral = gs.best_model.estimate_integral(N_INTEGRAL, SEED_INTEGRAL)

    fig = plot_density_2D(X_train, x_grid, y_grid, dx, dy, density_pred, density_true, 
                        title="", integral=integral, save_path=run_dir / "density_estimation.pdf")

    plt.close(fig)


def main(n_cores=1):
    print(f"\n{'='*70}\n[density]  solver={SOLVER}  n={N}  \n{'='*70}")
    try:
        run_one(N, SEED, n_cores)
    except Exception as e:
        print(f"FAILED [density]: solver={SOLVER}  n={N}    ({e})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=1)
    args = parser.parse_args()
    main(n_cores=args.n_cores)
