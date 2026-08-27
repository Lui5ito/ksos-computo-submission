import numpy as np
from dualforms.problems.utils.utils_kernels import make_kernel
from dualforms.problems.utils.utils_solver import solve_dispatch
from dualforms.problems.utils.utils_regularization import omega_star, Ahat_and_f
from dualforms.problems.utils.utils_predict import f_pred
from sklearn.preprocessing import StandardScaler

class Dual_MultiQuantile(object):
    """
    Joint non-crossing multi-quantile regression using kernel sum-of-squares with the sum of pinball losses.

    This class implements the joint quantile regression method described in the paper, 
    where the regression problem is formulated and solved through its dual 
    optimization problem (see Equation (19) in the paper).

    Parameters
    ----------
    quantile_levels: list
        Quantile levels to be estimated.
    median_kernel: callable
        Positive semi-definite kernel associated to the real-valued median function.
    quantile_kernels : list[callable]
        List of positive semi-definite kernels associated to each positive quantile incrementation functions.
    median_theta: object
        Lengthscales associated to median_kernel.
    quantile_thetas : list[object]
        List of lengthscales associated to the quantile_kernels.
    lambda_m: float
        Regularization parameter for the ridge regularization term.
    lambda1s : list[float]
        List of regularization parameters for the trace regularization terms.
    lambda2s : list[float]
        List of regularization parameters for the Frobenius-norm regularization terms.
    """

    def __init__(
        self,
        quantile_levels: list[float],
        median_kernel,
        quantile_kernels: list,
        median_theta,
        quantile_thetas: list,
        lambda_m: float,
        lambda1s: list[float],
        lambda2s: list[float],
    ):

        # kSoS problem definition
        self.median_kernel = median_kernel
        self.median_theta = median_theta
        self.lambda_m = lambda_m

        lower_idx = sorted([i for i, q in enumerate(quantile_levels) if q < 0.5], key=lambda i: -quantile_levels[i])
        upper_idx = sorted([i for i, q in enumerate(quantile_levels) if q > 0.5], key=lambda i: quantile_levels[i])

        def _reorder(lst, order):
            return [lst[i] for i in order]

        self.lower_quantile_levels = _reorder(quantile_levels, lower_idx)
        self.upper_quantile_levels = _reorder(quantile_levels, upper_idx)
        self.lower_kernels = _reorder(quantile_kernels, lower_idx)
        self.upper_kernels = _reorder(quantile_kernels, upper_idx)
        self.lower_thetas = _reorder(quantile_thetas, lower_idx)
        self.upper_thetas = _reorder(quantile_thetas, upper_idx)
        self.lower_lambda1 = _reorder(lambda1s, lower_idx)
        self.lower_lambda2 = _reorder(lambda2s, lower_idx)
        self.upper_lambda1 = _reorder(lambda1s, upper_idx)
        self.upper_lambda2 = _reorder(lambda2s, upper_idx)

        # Training sample variables
        self.n = None
        self.d = None
        self.X_train = None
        self.y_train = None

        # Optimization variables
        self.Ahat = None
        self.gammahat = None
        self.eta = None
        self.f = None
        self.optimal_variables = None

        # Solver variables
        self.solver_state = None
        self.solver_iter = None
        self.solver_time = None
        self.solver_success = None

    def dual_function(self, dual_vars, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train):
        n_lower = len(self.lower_quantile_levels)
        n_upper = len(self.upper_quantile_levels)

        alpha_0 = dual_vars[:self.n]
        Alpha_L = dual_vars[self.n : self.n * (1 + n_lower)].reshape(n_lower, self.n)
        Alpha_U = dual_vars[self.n * (1 + n_lower) :].reshape(n_upper, self.n)

        alpha_sum = alpha_0 + Alpha_L.sum(axis=0) + Alpha_U.sum(axis=0)

        inside_vars_L = -np.flipud(np.cumsum(np.flipud(Alpha_L), axis=0))
        inside_vars_U = np.flipud(np.cumsum(np.flipud(Alpha_U), axis=0))

        first_term = np.dot(y_train.flatten(), alpha_sum)
        second_term = (1 / (4 * lambda_m)) * np.dot(alpha_sum, gram_matrix_median @ alpha_sum)

        third_term = 0.0
        for Phi_k, lam1_k, lam2_k, iv_k in zip(chol_gram_matrices_lower, lower_lambda1, lower_lambda2, inside_vars_L):
            third_term += omega_star(iv_k, 0, lam1_k, lam2_k, Phi_k)
        for Phi_k, lam1_k, lam2_k, iv_k in zip(chol_gram_matrices_upper, upper_lambda1, upper_lambda2, inside_vars_U):
            third_term += omega_star(iv_k, 0, lam1_k, lam2_k, Phi_k)

        return -(first_term - second_term - third_term)

    def grad_dual_function(self, dual_vars, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train):
        n_lower = len(self.lower_quantile_levels)
        n_upper = len(self.upper_quantile_levels)

        alpha_0 = dual_vars[:self.n]
        Alpha_L = dual_vars[self.n : self.n * (1 + n_lower)].reshape(n_lower, self.n)
        Alpha_U = dual_vars[self.n * (1 + n_lower) :].reshape(n_upper, self.n)
        alpha_sum = alpha_0 + Alpha_L.sum(axis=0) + Alpha_U.sum(axis=0)

        _, fs_lower, _, fs_upper = self._compute_Ahat_and_f(
            dual_vars, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, chol_gram_matrices_lower, chol_gram_matrices_upper
        )

        y = y_train.flatten()
        common = y - (1 / (2 * lambda_m)) * (gram_matrix_median @ alpha_sum)

        grad_alpha_0 = common

        grad_Alpha_L = np.zeros((n_lower, self.n))
        if n_lower > 0:
            cumfs_lower = np.cumsum(np.stack(fs_lower), axis=0)
            for k in range(n_lower):
                grad_Alpha_L[k] = common + cumfs_lower[k]

        grad_Alpha_U = np.zeros((n_upper, self.n))
        if n_upper > 0:
            cumfs_upper = np.cumsum(np.stack(fs_upper), axis=0)
            for k in range(n_upper):
                grad_Alpha_U[k] = common - cumfs_upper[k]

        return -np.concatenate([grad_alpha_0, grad_Alpha_L.reshape(-1), grad_Alpha_U.reshape(-1)])

    def _compute_Ahat_and_f(self, dual_vars, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, chol_gram_matrices_lower, chol_gram_matrices_upper):
        n_lower = len(self.lower_quantile_levels)
        n_upper = len(self.upper_quantile_levels)

        Alpha_L = dual_vars[self.n : self.n * (1 + n_lower)].reshape(n_lower, self.n)
        Alpha_U = dual_vars[self.n * (1 + n_lower) :].reshape(n_upper, self.n)

        inside_vars_L = -np.flipud(np.cumsum(np.flipud(Alpha_L), axis=0))
        inside_vars_U = np.flipud(np.cumsum(np.flipud(Alpha_U), axis=0))

        Ahats_lower, fs_lower = [], []
        for Phi_k, lam1_k, lam2_k, iv_k in zip(chol_gram_matrices_lower, lower_lambda1, lower_lambda2, inside_vars_L):
            A_k, f_k = Ahat_and_f(iv_k, 0, lam1_k, lam2_k, Phi_k)
            Ahats_lower.append(A_k)
            fs_lower.append(f_k)

        Ahats_upper, fs_upper = [], []
        for Phi_k, lam1_k, lam2_k, iv_k in zip(chol_gram_matrices_upper, upper_lambda1, upper_lambda2, inside_vars_U):
            A_k, f_k = Ahat_and_f(iv_k, 0, lam1_k, lam2_k, Phi_k)
            Ahats_upper.append(A_k)
            fs_upper.append(f_k)

        return Ahats_lower, fs_lower, Ahats_upper, fs_upper

    def duality_gap(self, dual_vars, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train):
        """
        Compute primal objective function from dual optimization variables and resulting duality gap.
        """
        n_lower = len(self.lower_quantile_levels)
        n_upper = len(self.upper_quantile_levels)
        alpha_0 = dual_vars[:self.n]
        Alpha_L = dual_vars[self.n : self.n * (1 + n_lower)].reshape(n_lower, self.n)
        Alpha_U = dual_vars[self.n * (1 + n_lower) :].reshape(n_upper, self.n)
        alpha_sum = alpha_0 + Alpha_L.sum(axis=0) + Alpha_U.sum(axis=0)
        median = (1 / (2 * lambda_m)) * (gram_matrix_median @ alpha_sum)
        y = y_train.flatten()

        Ahats_lower, fs_lower, Ahats_upper, fs_upper = self._compute_Ahat_and_f(
            dual_vars, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, chol_gram_matrices_lower, chol_gram_matrices_upper
        )

        primal = 0.5 * np.sum(np.abs(y - median))
        primal += (1 / (4 * lambda_m)) * np.dot(alpha_sum, gram_matrix_median @ alpha_sum)

        if fs_lower:
            cumfs_lower = np.cumsum(np.stack(fs_lower), axis=0)
            q_lower = median[np.newaxis, :] - cumfs_lower
            residuals_lower = y[np.newaxis, :] - q_lower
            tau_lower = np.array(self.lower_quantile_levels)[:, np.newaxis]
            primal += np.sum(tau_lower * np.maximum(residuals_lower, 0) + (1 - tau_lower) * np.maximum(-residuals_lower, 0))

        if fs_upper:
            cumfs_upper = np.cumsum(np.stack(fs_upper), axis=0)
            q_upper = median[np.newaxis, :] + cumfs_upper
            residuals_upper = y[np.newaxis, :] - q_upper
            tau_upper = np.array(self.upper_quantile_levels)[:, np.newaxis]
            primal += np.sum(tau_upper * np.maximum(residuals_upper, 0) + (1 - tau_upper) * np.maximum(-residuals_upper, 0))

        for A_k, lam1_k, lam2_k in zip(Ahats_lower, lower_lambda1, lower_lambda2):
            primal += lam1_k * np.trace(A_k) + lam2_k * np.linalg.norm(A_k, 'fro') ** 2
        for A_k, lam1_k, lam2_k in zip(Ahats_upper, upper_lambda1, upper_lambda2):
            primal += lam1_k * np.trace(A_k) + lam2_k * np.linalg.norm(A_k, 'fro') ** 2

        self.primal_value = primal

        dual = -self.dual_function(dual_vars, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train)
        self.dual_value = dual
        if dual == 0:
            return np.inf

        return np.abs((primal - dual) / dual).item()

    def check_convergence(self, dual_vars, gap_tol, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train):
        gap = self.duality_gap(dual_vars, lambda_m, lower_lambda1, lower_lambda2, upper_lambda1, upper_lambda2, gram_matrix_median, chol_gram_matrices_lower, chol_gram_matrices_upper, y_train)
        
        return gap < gap_tol

    def cv_score(self, X, y):
        """
        Sum of pinball losses score.
        """
        predictions = self.predict(X)
        
        loss = 0.0

        for tau, preds in predictions.items():
            residuals = y.flatten() - preds.flatten()
            loss += np.sum(tau * np.maximum(residuals, 0) + (1 - tau) * np.maximum(-residuals, 0))
        
        return loss

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        solver,
        max_iters: int,
        gap_tol: float,
        use_low_rank: bool,
        target_variance: float,
        method_low_rank: str = "svd",
    ):
        """
        Train the joint non-crossing multi-quantiles regression model on the given training data.

        The input data are standardized before fitting. The method constructs
        the kernel matrices, solves the dual optimization problem, 
        checks the convergence of the dual to the primal and stores
        the resulting optimal parameters in the instance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature vectors in training data.
        y : np.ndarray of shape (n_samples, 1)
            Target values in training data.
        solver : str
            Optimization solver used to solve the dual problem. Supported
            values are ``"BFGS"`` and ``"AGD"``.
        max_iters : int
            Maximum number of solver iterations.
        gap_tol : float
            Convergence tolerance for the duality gap.
        use_low_rank : bool
            Whether to use a low-rank approximation on the kernel matrix.
        target_variance : float
            Target fraction of variance retained by the low-rank approximation.
        method_low_rank : str, default="svd"
            Low-rank approximation method. Supported values are ``"svd"``,
            ``"rsvd"``, and ``"nystrom"``.

        Returns
        -------
        None
            The fitted model and optimization results are stored in the
            instance attributes.
        """
        # Scale X_train
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        self.X_train = X
        self.y_train = y
        self.n = self.X_train.shape[0]
        self.d = self.X_train.shape[1]

        self.use_low_rank = use_low_rank
        self.target_variance = target_variance
        self.method_low_rank = method_low_rank

        # Assemble kernel matrices on training sample
        self.gram_matrix_median, _, _, self.median_kernel, self.median_theta, distances = make_kernel(self.X_train, self.median_theta, self.median_kernel, False, None, None, None)

        results_lower = [make_kernel(self.X_train, t, k, self.use_low_rank, self.method_low_rank, self.target_variance, distances) for t, k in zip(self.lower_thetas, self.lower_kernels)]
        self.lower_kernels = [k for _, _, _, k, _, _ in results_lower]
        self.lower_thetas = [t for _, _, _, _, t, _ in results_lower]
        self.gram_matrices_lower = [g for g, _, _, _, _, _ in results_lower]
        self.chol_gram_matrices_lower = [c for _, c, _, _, _, _ in results_lower]
        self.chol_gram_matrices_lower_inference = [ci for _, _, ci, _, _, _ in results_lower]
        self.phi_gram_matrices_lower = self.chol_gram_matrices_lower

        results_upper = [make_kernel(self.X_train, t, k, self.use_low_rank, self.method_low_rank, self.target_variance, distances) for t, k in zip(self.upper_thetas, self.upper_kernels)]
        self.upper_kernels = [k for _, _, _, k, _, _ in results_upper]
        self.upper_thetas = [t for _, _, _, _, t, _ in results_upper]
        self.gram_matrices_upper = [g for g, _, _, _, _, _ in results_upper]
        self.chol_gram_matrices_upper = [c for _, c, _, _, _, _ in results_upper]
        self.chol_gram_matrices_upper_inference = [ci for _, _, ci, _, _, _ in results_upper]
        self.phi_gram_matrices_upper = self.chol_gram_matrices_upper

        # Solve dual optimization problem
        dual_dim = (1 + len(self.lower_quantile_levels) + len(self.upper_quantile_levels)) * self.n
        initial_value = np.zeros(dual_dim)

        median_bounds = ((-0.5, 0.5),) * self.n
        lower_bounds = tuple(
            (-(1 - tau), tau)
            for tau in self.lower_quantile_levels
            for _ in range(self.n)
        )
        upper_bounds = tuple(
            (-(1 - tau), tau)
            for tau in self.upper_quantile_levels
            for _ in range(self.n)
        )
        bounds = median_bounds + lower_bounds + upper_bounds

        solver_tols = (gap_tol,)
        solver_args = (self.lambda_m, self.lower_lambda1, self.lower_lambda2, self.upper_lambda1, self.upper_lambda2, self.gram_matrix_median, self.chol_gram_matrices_lower, self.chol_gram_matrices_upper, self.y_train)

        res = solve_dispatch(
            solver_name=solver,
            initial_value=initial_value,
            bounds=bounds,
            obj_func=self.dual_function,
            obj_grad=self.grad_dual_function,
            gap_func=self.check_convergence,
            max_iters=max_iters,
            solver_tols=solver_tols,
            solver_args=solver_args,
        )
        self.optimal_variables = res.x

        # Convergence check
        self.solver_success = res.solver_success
        self.solver_iter = res.nit
        self.solver_state = res.message
        tag = "[OPTIM: CONVERGED]" if self.solver_success else "[OPTIM: FAILED]"
        print(f"{tag} Solver={solver} | Iterations={self.solver_iter}")

        # Retrieve optimal solution
        self.Ahats_lower, self.fs_lower, self.Ahats_upper, self.fs_upper = self._compute_Ahat_and_f(
            self.optimal_variables,
            self.lower_lambda1, self.lower_lambda2, self.upper_lambda1, self.upper_lambda2,
            self.chol_gram_matrices_lower, self.chol_gram_matrices_upper,
        )
        alpha_0 = self.optimal_variables[:self.n]
        Alpha_L = self.optimal_variables[self.n : self.n * (1 + len(self.lower_quantile_levels))].reshape(len(self.lower_quantile_levels), self.n)
        Alpha_U = self.optimal_variables[self.n * (1 + len(self.lower_quantile_levels)) :].reshape(len(self.upper_quantile_levels), self.n)
        alpha_sum = alpha_0 + Alpha_L.sum(axis=0) + Alpha_U.sum(axis=0)
        self.median = (1 / (2 * self.lambda_m)) * (self.gram_matrix_median.T @ alpha_sum)

    def predict(self, X: np.ndarray, chunk_size: int = 1000) -> dict:
        """
        Predict quantile functions (including the median) for input X, processed in chunks to save memory.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data where to predict the quantiles.
        chunk_size : int
            Number of samples to process at a time.

        Returns
        -------
        dict
            A dictionary where keys are quantile levels and values are the predicted quantiles (including the median).
        """
        n_samples = X.shape[0]

        all_taus = [0.5] + self.lower_quantile_levels + self.upper_quantile_levels
        results = {tau: np.empty(n_samples) for tau in all_taus}

        n_lower = len(self.lower_quantile_levels)
        n_upper = len(self.upper_quantile_levels)

        alpha_0 = self.optimal_variables[:self.n]
        Alpha_L = self.optimal_variables[self.n : self.n * (1 + n_lower)].reshape(n_lower, self.n)
        Alpha_U = self.optimal_variables[self.n * (1 + n_lower) :].reshape(n_upper, self.n)
        alpha_sum = alpha_0 + Alpha_L.sum(axis=0) + Alpha_U.sum(axis=0)

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]

            X_test_chunk = self.scaler.transform(X_chunk)


            k_train_test_median_chunk = self.median_kernel(self.X_train, X_test_chunk)
            results[0.5][start_idx:end_idx] = (1 / (2 * self.lambda_m)) * (k_train_test_median_chunk.T @ alpha_sum)

            fs_lower_test_chunk = [f_pred(self.X_train, X_test_chunk, kernel, A_k, Phi_k, Phi_k_inf, self.use_low_rank, self.method_low_rank) for kernel, A_k, Phi_k, Phi_k_inf in zip(self.lower_kernels, self.Ahats_lower, self.chol_gram_matrices_lower, self.chol_gram_matrices_lower_inference)]
            fs_upper_test_chunk = [f_pred(self.X_train, X_test_chunk, kernel, A_k, Phi_k, Phi_k_inf, self.use_low_rank, self.method_low_rank) for kernel, A_k, Phi_k, Phi_k_inf in zip(self.upper_kernels, self.Ahats_upper, self.chol_gram_matrices_upper, self.chol_gram_matrices_upper_inference)]

            if fs_lower_test_chunk:
                cumfs_lower_chunk = np.cumsum(np.stack(fs_lower_test_chunk), axis=0)
                for k, tau in enumerate(self.lower_quantile_levels):
                    results[tau][start_idx:end_idx] = results[0.5][start_idx:end_idx] - cumfs_lower_chunk[k]

            if fs_upper_test_chunk:
                cumfs_upper_chunk = np.cumsum(np.stack(fs_upper_test_chunk), axis=0)
                for k, tau in enumerate(self.upper_quantile_levels):
                    results[tau][start_idx:end_idx] = results[0.5][start_idx:end_idx] + cumfs_upper_chunk[k]

        return results