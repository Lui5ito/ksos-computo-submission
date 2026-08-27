import numpy as np
from scipy.linalg import eigh
from scipy.linalg import eigvalsh

def compute_interior_matrix(alpha, U, lambda_1, chol):
    """
    Computation of the matrix which appears in the positive part expression of the dual problem in Equation (11).
    """
    alpha = alpha.flatten()
    scaled_chol = chol * alpha
    interior_matrix = scaled_chol @ chol.T
    np.fill_diagonal(interior_matrix, interior_matrix.diagonal() - lambda_1)
    if not np.isscalar(U) or U != 0:
        interior_matrix += U

    return interior_matrix

def omega_star(alpha, U, lambda_1, lambda_2, chol):
    """
    Fenchel conjugate of elastic-net regularization Omega^+, see Equation (21).
    """
    interior = compute_interior_matrix(alpha, U, lambda_1, chol)
    try:
        evals = eigvalsh(
            interior,
            driver='evd',
            check_finite=False,
            overwrite_a=False
        )
    except Exception:
        evals = eigvalsh(
            interior,
            driver='ev',
            check_finite=False,
            overwrite_a=False
        )
    evals_t = np.maximum(evals, 0)
    norm_sq = np.vdot(evals_t, evals_t)
    return norm_sq / (4 * lambda_2)

def grad_omega_star(alpha, U, lambda_1, lambda_2, chol):
    """
    Gradient of Fenchel conjugate of elastic-net regularization Omega^+, see Equation (22).
    """
    interior = compute_interior_matrix(alpha, U, lambda_1, chol)
    try:
        evals, evecs = eigh(
            interior,
            driver='evd',
            check_finite=False,
            overwrite_a=False
        )
    except Exception:
        evals, evecs = eigh(
            interior,
            driver='ev',
            check_finite=False,
            overwrite_a=False
        )
    evals_t = np.maximum(evals, 0)
    B_plus = (evecs * evals_t) @ evecs.T
    return B_plus / (2 * lambda_2)

def Ahat_and_f(inside_var, U, lambda_1, lambda_2, chol):
    """
    Computation of PSD operator A and associated kSoS function f on training sample (Equation (13)).
    The predictions are clipped due to possible numerical errors.
    """
    Ahat = grad_omega_star(inside_var, U, lambda_1, lambda_2, chol)
    f = np.einsum('ji,jk,ki->i', chol, Ahat, chol, optimize=True)
    f_clipped = np.maximum(f, 1e-12)
    return Ahat, f_clipped