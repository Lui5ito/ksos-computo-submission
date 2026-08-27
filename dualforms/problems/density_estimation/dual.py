import numpy as np
from dualforms.problems.utils.utils_kernels import make_kernel
from dualforms.problems.utils.utils_solver import solve_dispatch
from dualforms.problems.utils.utils_regularization import omega_star, Ahat_and_f
from dualforms.problems.utils.utils_predict import f_pred
from sklearn.preprocessing import StandardScaler
from scipy.stats import uniform, multivariate_normal, qmc

class Dual_Density(object):
    """
    Density estimation using kernel sum-of-squares with negative log-likelihood loss.

    This class implements the density estimation method described in the paper, 
    where the density estimation problem is formulated and solved through its dual 
    optimization problem (see Equation (16) in the paper).

    Parameters
    ----------
    kernel : callable
        Positive semi-definite kernel associated to the positive parameterization function of the density.
    theta : object
        Lengthscales associated to the kernel.
    lambda1 : float
        Regularization parameter for the trace regularization term.
    lambda2 : float
        Regularization parameter for the Frobenius-norm regularization term.
    """

    def __init__(
        self,
        kernel,
        theta,
        lambda1: float,
        lambda2: float,
    ):
        # kSoS problem definition
        self.kernel = kernel
        self.theta = theta
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.W_nu = None

        # Training sample variables
        self.n = None
        self.d = None
        self.X_train = None

        # Optimization variables
        self.Ahat = None
        self.f = None
        self.optimal_variables = None

        # Solver variables
        self.solver_state = None
        self.solver_iter = None
        self.solver_time = None
        self.solver_success = None

    def dual_function(self, dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu):
        theta = dual_vars[0]
        alpha = np.exp(dual_vars[1:self.n+2])

        second_term = np.sum(dual_vars[1:self.n+2])
        third_term = omega_star(alpha, -theta*W_nu, lambda1, lambda2, chol_gram_matrix)

        return - (-theta + self.n + second_term - third_term)

    def grad_dual_function(self, dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu):
        alpha = np.exp(dual_vars[1:self.n+2])

        A, f = self._compute_Ahat_and_f(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)
        
        grad_wrt_theta = np.array([-1 + np.sum(A*W_nu)]).reshape(-1, 1)
        grad_wrt_alpha = ((1/alpha).reshape(-1, 1) - f) * alpha.reshape(-1, 1)

        return -np.concatenate([grad_wrt_theta, grad_wrt_alpha])

    def _compute_Ahat_and_f(self, dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu):
        theta = dual_vars[0]
        alpha = np.exp(dual_vars[1:self.n+2])

        A, f = Ahat_and_f(alpha, -theta*W_nu, lambda1, lambda2, chol_gram_matrix)

        return A, f.reshape(-1, 1)

    def _compute_p(self, X):
        match self.ref_density:
            case "uniform":
                p = np.prod(uniform.pdf(X, loc=-6, scale=12), axis=1)
            case "gaussian":
                p = multivariate_normal.pdf(X, np.zeros(self.d), np.eye(self.d))
            case _:
                raise ValueError("Reference density not implemented.", self.ref_density)

        return p

    def constraints(self, dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu):
        Ahat, _ = self._compute_Ahat_and_f(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)

        return np.abs(np.sum(Ahat*W_nu) - 1)

    def duality_gap(self, dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu):
        """
        Compute primal objective function from dual optimization variables and resulting duality gap.
        """
        Ahat, f = self._compute_Ahat_and_f(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)

        primal = - np.sum(np.log(f))
        primal += self.lambda1 * np.trace(Ahat)
        primal += self.lambda2 * (np.linalg.norm(Ahat, "fro") ** 2)

        dual = - self.dual_function(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)

        if dual == 0:
            return np.inf

        return np.abs((primal - dual) / dual).item()

    def cv_score(self, X):
        """
        L2 score estimated with quasi Monte-Carlo.
        """
        preds = self.predict(X)
        sampler = qmc.Sobol(d=X.shape[1], scramble=False)
        sample = -2 + 4 * sampler.random_base2(m=13)
        integral_dens_squared = np.mean(self.predict(sample)**2) * 16

        return integral_dens_squared - 2 * np.mean(preds)

    def check_convergence(self, dual_vars, cov_tol, gap_tol, lambda1, lambda2, chol_gram_matrix, W_nu):
        constraints = self.constraints(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)
        gap = self.duality_gap(dual_vars, lambda1, lambda2, chol_gram_matrix, W_nu)

        if constraints < cov_tol:
            return (gap < gap_tol)

        return False

    def estimate_integral(self, n_samples, seed):
        rng = np.random.default_rng(seed=seed)
        match self.ref_density:
            case "uniform":
                U = rng.uniform(-6, 6, (n_samples, self.d))
            case "gaussian":
                U = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), n_samples)
            case _:
                raise ValueError("Reference density not implemented.", self.ref_density)

        weight = 1 / self._compute_p(U)
        U = self.scaler.inverse_transform(U)
        preds = self.predict(U)
        det_jacobian = np.prod(1 / self.scaler.scale_)

        return np.mean(preds * weight / det_jacobian)

    def train(
        self,
        X: np.ndarray,
        ref_density: str,
        n_mc_density: int,
        solver,
        max_iters: int,
        cov_tol: float,
        gap_tol: float,
        use_low_rank: bool,
        target_variance: float,
        method_low_rank: str = "svd",
    ):
        """
        Train the density estimation model on the given training data.

        The input data are standardized before fitting. The method constructs
        the kernel matrices, approximates the reference-density term using
        Monte Carlo sampling, solves the dual optimization problem, checks the convergence
        of the dual to the primal and stores the resulting optimal parameters in the instance.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Feature vectors in training data.
        ref_density : str
            Reference density used for Monte Carlo integration. Supported
            values are ``"uniform"`` (uniform distribution on ``[-6, 6]^d``)
            and ``"gaussian"`` (standard multivariate normal distribution).
        n_mc_density : int
            Number of Monte-Carlo samples used to approximate the reference
            density term.
        solver : str
            Optimization solver used to solve the dual problem. 
            Supported values are ``"BFGS"`` and ``"AGD"``.
        max_iters : int
            Maximum number of solver iterations.
        cov_tol : float
            Convergence tolerance for the constraint.
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
        self.n = self.X_train.shape[0]
        self.d = self.X_train.shape[1]

        self.ref_density = ref_density

        self.use_low_rank = use_low_rank
        self.target_variance = target_variance
        self.method_low_rank = method_low_rank

        # Assemble kernel matrices on training sample
        self.gram_matrix, self.chol_gram_matrix, self.chol_gram_matrix_inference, self.kernel, self.theta, _,  = make_kernel(self.X_train, self.theta, self.kernel, self.use_low_rank, self.method_low_rank, self.target_variance, None)
        self.phi_gram_matrix = self.chol_gram_matrix

        # Compute W_nu by Monte-Carlo approximation with Equation (17)
        rng = np.random.default_rng(seed=42)
        match ref_density:
            case "uniform":
                U = rng.uniform(-6, 6, (n_mc_density, self.d))
            case "gaussian":
                U = rng.multivariate_normal(np.zeros(self.d), np.eye(self.d), n_mc_density)
            case _:
                raise ValueError("Reference density not implemented.", ref_density)

        K_mc = self.kernel(X, U)
        match self.use_low_rank:
            case False:
                A_mc = np.linalg.solve(self.chol_gram_matrix.T, K_mc)
            case True:
                match method_low_rank:
                    case "svd" | "rsvd":
                        A_mc = self.chol_gram_matrix_inference @ K_mc
                    case "nystrom":
                        A_mc = np.linalg.solve(self.chol_gram_matrix.T, K_mc)
        self.W_nu = (1/n_mc_density) * (A_mc @ A_mc.T)

        # Solve dual optimization problem
        initial_value = np.zeros(self.n+1)

        bounds = ((None, None),) * (1 + self.n)

        solver_tols = (cov_tol, gap_tol)
        solver_args = (self.lambda1, self.lambda2, self.chol_gram_matrix, self.W_nu)

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
        self.Ahat, self.f = self._compute_Ahat_and_f(self.optimal_variables, self.lambda1, self.lambda2, self.chol_gram_matrix, self.W_nu)

    def predict(self, X: np.ndarray, chunk_size: int = 1000):
        """
        Predict density function for input X, processed in chunks to save memory.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data where to predict the density.
        chunk_size : int
            Number of samples to process at a time.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted density values at each input sample.
        """
        n_samples = X.shape[0]
        results = np.empty(n_samples)
        
        # Calculate this constant once outside the loop
        det_jacobian = np.prod(1 / self.scaler.scale_)
        
        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]

            X_test_chunk = self.scaler.transform(X_chunk)

            f_test_chunk = f_pred(
                self.X_train, X_test_chunk, self.kernel, self.Ahat,
                self.chol_gram_matrix, self.chol_gram_matrix_inference,
                self.use_low_rank, self.method_low_rank,
            )
            p_chunk = self._compute_p(X_test_chunk)

            results[start_idx:end_idx] = f_test_chunk * p_chunk * det_jacobian

        return results