import numpy as np
from scipy.stats import norm
from scipy.spatial.distance import pdist
from scipy.linalg import svd
from sklearn.utils.extmath import randomized_svd as _sklearn_rsvd
import time


def cholesky_decomposition(M, upper: bool = True, verbose: bool = False):
    """
    Perform Cholesky decomposition, adding a small penalisation if it fails.
    Returns **upper** Cholesky decomposition matrix.
    """
    try:
        return np.linalg.cholesky(M, upper=upper)
    except np.linalg.LinAlgError:
        pen_values = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
        for pen in pen_values:
            try:
                V = np.linalg.cholesky(M + pen * np.eye(M.shape[0]), upper=upper)
                if verbose:
                    print(f'Choleksy failed, using a penalisation of {pen}.')
                return V
            except np.linalg.LinAlgError as e:
                if pen == pen_values[-1]:
                    raise RuntimeError(
                        f'An error occurred during training with penalisation={pen}: {e}'
                    )


def find_truncated_rank(M, target_variance=0.995, verbose=True, method="svd"):
    """
    Compute low-rank approximation of matrix M with specified target fraction of variance retained,
    with SVD or randomized SVD. Best rank is estimated in an adaptive way for randomized SVD.
    """
    m_opt = M.shape[0]
    match method:
        case "svd":
            U, s, Vt = svd(M, full_matrices=True, compute_uv=True, overwrite_a=False, check_finite=False, lapack_driver='gesdd')
            cum_var = np.cumsum(s**2) / np.sum(s**2)
            idx = np.searchsorted(cum_var, target_variance)

            # 1. Find the smallest rank m to reach target variance
            if idx < len(s):
                m_opt = idx + 1

            # 2. Fallback: no truncation reached the target, we perform full SVD
            else:
                m_opt = M.shape[0]

        case "rsvd":
            m_to_test = range(50, M.shape[0]+1, 50)
            M_norm_sq = np.linalg.norm(M, "fro")**2
            for m in m_to_test:
                U, s, Vt = _sklearn_rsvd(M, n_components=m, random_state=42)
                cum_var = np.cumsum(s**2) / M_norm_sq
                idx = np.searchsorted(cum_var, target_variance)
                # 1. Find the smallest rank m to reach target variance
                if idx < len(s):
                    m_opt = idx + 1
                    break

    U_opt  = U[:, :m_opt]
    s_opt  = s[:m_opt]
    Vt_opt = Vt[:m_opt, :]
    if verbose:
        explained = cum_var[idx] * 100
        print(f"Selected rank m_opt = {m_opt} explaining {explained:.2f}% variance.")

    return U_opt, s_opt, Vt_opt


def nystrom(K, r, seed):
    """
    Compute Nystrom low-rank approximation of matrix K with rank r.
    """
    n = K.shape[0]
 
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
 
    indices = rng.choice(n, size=r, replace=False)
    indices = np.sort(indices)
    K_mm = K[np.ix_(indices, indices)]
    K_nm = K[:, indices]
    eigvals, eigvecs = np.linalg.eigh(K_mm)
    eigvals = np.maximum(eigvals, 0.0)
    tol = eigvals.max() * r * np.finfo(float).eps
    inv_sqrt = np.where(eigvals > tol,
                        1.0 / np.sqrt(np.maximum(eigvals, tol)), 0.0)
    W = K_nm @ (eigvecs * inv_sqrt) 
    V_approx = W.T
    K_approx = W @ W.T
 
    elapsed = time.perf_counter() - t0

    diff = K - K_approx
    explained = 1.0 - norm(diff, "fro") ** 2 / norm(K, "fro") ** 2
    
    return V_approx, K_approx, indices, elapsed, explained


def nystrom_adaptive_rank(K, target_variance, r_min, r_max, seed, verbose=True):
    """
    Compute low-rank approximation of matrix K with specified target fraction of variance retained,
    with Nystrom approximation. 
    Best rank is estimated in an adaptive way with binary search between r_min and r_max.
    """
    n = K.shape[0]
    # Binary search
    lo, hi = r_min, r_max
    search_log = []

    V_approx, K_approx, indices, t_hi, explained_hi = nystrom(K, hi, seed)
    search_log.append((hi, explained_hi, t_hi))
    if explained_hi < target_variance:
        if verbose:
            print(f"Warning: even r={hi} gives explained variance {explained_hi:.4e} < target {target_variance:.4e}.")
            print("Returning r_max.")
        best_r = hi
        final_explained = explained_hi
        final_elapsed = t_hi
        
    else:
        V_approx, K_approx, indices, t_lo, explained_lo = nystrom(K, lo, seed)
        search_log.append((lo, explained_lo, t_lo))
        if explained_lo >= target_variance:
            best_r = lo
        else:
            while lo < hi - 1:
                mid = (lo + hi) // 2
                V_approx, K_approx, indices, t_mid, explained_mid = nystrom(K, mid, seed)
                search_log.append((mid, explained_mid, t_mid))
                if verbose:
                    print(f"  Binary search: r={mid:4d}  explained variance={explained_mid:.4e}  time={t_mid*1e3:.2f} ms")
                if explained_mid >= target_variance:
                    hi = mid
                else:
                    lo = mid
            best_r = hi
 
        V_approx, K_approx, indices, final_elapsed, final_explained = nystrom(K, best_r, seed)
 
    if verbose:
        print(f"\nAdaptive Nyström  (n={n}, target variance={target_variance:.2e})")
        print(f"  Selected rank    : {best_r}")
        print(f"  Achieved error   : {final_explained:.4e}")
        print(f"  Compression ratio: {best_r/n:.2%} of n")
        print(f"  Time (final run) : {final_elapsed:.2f} ms")
 
    search_log.sort(key=lambda x: x[0])
    
    return best_r, V_approx, K_approx, search_log


def make_gram(X_train, kernel, use_low_rank, method_low_rank, target_variance):
    """
    Compute Gram matrix on training sample, possibly with low-rank approximation with specified target fraction of variance retained,
    with SVD, randomized SVD or Nystrom approximation. 
    """
    gram = kernel(X=X_train)
    chol_inference = None
    match use_low_rank:
        case False:
            chol = cholesky_decomposition(M=gram, upper=True)
        case True:
            match method_low_rank:
                case "svd" | "rsvd":
                    _, s, Vt = find_truncated_rank(M=gram, target_variance=target_variance, verbose=False, method=method_low_rank)
                    chol = np.diag(np.sqrt(s)) @ Vt
                    chol_inference = np.diag(1 / np.sqrt(s)) @ Vt
                    gram = chol.T @ chol
                case "nystrom":
                    _, V_approx, _, _ = nystrom_adaptive_rank(K=gram, target_variance=target_variance, r_min=10, r_max=n, seed=42, verbose=False)
                    chol = V_approx
                    gram = V_approx.T @ V_approx
                case _:
                    raise ValueError("Low rank method not implemented yet", method_low_rank)

    return gram, chol, chol_inference


def make_kernel(X_train, theta, kernel, use_low_rank=None, method_low_rank=None, target_variance=None, distances=None):
    """
    Compute useful kernel matrices on training sample depending on the chosen kernel and its lengthscales theta, 
    possibly with low-rank approximation with specified target fraction of variance retained 
    with SVD, randomized SVD or Nystrom approximation. 
    """
    if theta == "ROT":
        if distances is None:
            distances = pdist(X_train, metric="euclidean")
        theta = np.median(distances)

    kernel.update(new_lengthscales=theta)
    gram, chol, chol_inference = make_gram(X_train, kernel, use_low_rank, method_low_rank, target_variance)

    return gram, chol, chol_inference, kernel, theta, distances 
