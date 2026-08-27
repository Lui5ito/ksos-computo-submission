"""Reproduce Figure 3 from the paper."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from dualforms.problems.density_estimation.dual import Dual_Density
from dualforms.kernels.matern import Matern
from dualforms.plots.density2D import plot_density_2D
from dualforms.data_generation.unsupervised import Unsupervised

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "main"

# ── kSoS problem settings ──────────────────────────────────────────────────────────────
N = 2000
SEED = 7
SOLVER = "BFGS"

MAX_ITERS = 100_000
COV_TOL = 1e-2
GAP_TOL = 1e-2
REF_DENSITY = "gaussian"
N_MC_DENSITY = 10_000
USE_LOW_RANK = True
TARGET_VARIANCE = 0.995
METHOD_LOW_RANK = "svd"

N_INTEGRAL = 10_000
SEED_INTEGRAL = 1

# ── Best hyperparameter combination identified by cross-validation (see appendix.py) ───
PARAMS = {
    "theta_x1": 0.7,
    "theta_x2": 0.8,
    "lambda1": 1.0,
    "lambda2": 1.0,
}


def build_model(theta_x1, theta_x2, lambda1, lambda2):
    return Dual_Density(kernel=Matern(), theta=[theta_x1, theta_x2], lambda1=lambda1, lambda2=lambda2)


def run_one(n, seed):
    """
    For a fixed training size and seed, generate the training sample,
    run a single optimization of the density estimation at fixed hyperparameters,
    evaluate the true density and plot the true density versus estimation.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = Unsupervised(name="banana")
    X_train = dataset.sample(n, seed)

    model = build_model(**PARAMS)
    model.train(
        X_train,
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

    sample_min = X_train.min(axis=0)
    sample_max = X_train.max(axis=0)
    x_grid, y_grid, dx, dy = dataset.sample_grid(n=200, sample_min=sample_min, sample_max=sample_max)

    density_pred = model.predict(np.vstack([x_grid.ravel(), y_grid.ravel()]).T).reshape(x_grid.shape)
    density_pred = density_pred / (np.sum(density_pred) * dx * dy)

    density_true = dataset.oracle(x_grid, y_grid, dx, dy)

    integral = model.estimate_integral(N_INTEGRAL, SEED_INTEGRAL)

    fig = plot_density_2D(X_train, x_grid, y_grid, dx, dy, density_pred, density_true,
                        title="", integral=integral, save_path=OUT_DIR / "density_estimation_main.pdf")
    plt.close(fig)


def main():
    print(f"\n{'='*70}\n[density]  solver={SOLVER}  n={N}  \n{'='*70}")
    try:
        run_one(N, SEED)
    except Exception as e:
        print(f"FAILED [density]: solver={SOLVER}  n={N}    ({e})")


if __name__ == "__main__":
    main()
