import numpy as np
from dualforms.problems.utils.utils_kernels import make_kernel
from dualforms.problems.utils.utils_solver import solve_dispatch
from dualforms.problems.utils.utils_regularization import omega_star, Ahat_and_f
from dualforms.problems.utils.utils_predict import f_pred
from sklearn.preprocessing import StandardScaler

class Dual_Heteroscedastic(object):
    """
    Joint heteroscedastic regression using kernel sum-of-squares with the natural parameterization and the negative log-likelihood loss.

    This class implements the joint heteroscedastic regression method described in the paper, 
    where the regression problem is formulated and solved through its dual 
    optimization problem (see Equation (18) in the paper).

    Parameters
    ----------
    mean_kernel : callable
        Positive semi-definite kernel associated to the real-valued parameterization mean/variance function.
    variance_kernel : callable
        Positive semi-definite kernel associated to the positive parameterization function of the variance (1/variance).
    mean_theta : object
        Lengthscales associated to mean_kernel.
    variance_theta : object
        Lengthscales associated to variance_kernel.
    lambda_m : float
        Regularization parameter for the ridge regularization term.
    lambda1 : float
        Regularization parameter for the trace regularization term.
    lambda2 : float
        Regularization parameter for the Frobenius-norm regularization term.
    """

    def __init__(
        self,
        mean_kernel,
        variance_kernel,
        mean_theta,
        variance_theta,
        lambda_m: float,
        lambda1: float,
        lambda2: float,
    ):
        # kSoS problem definition
        self.mean_kernel = mean_kernel
        self.variance_kernel = variance_kernel
        self.mean_theta = mean_theta
        self.variance_theta = variance_theta
        self.lambda_m = lambda_m
        self.lambda1 = lambda1
        self.lambda2 = lambda2

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

    def dual_function(self, dual_vars, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train):
        alpha = dual_vars[:self.n]
        z = dual_vars[self.n:]
        inside_var = np.exp(z) + (1/4) * alpha**2 - alpha*y_train.flatten()

        first_term =  self.n + np.sum(z)
        second_term = (1/(4*lambda_m)) * np.dot(alpha.T, gram_matrix_mean @ alpha)
        third_term = omega_star(inside_var, 0, lambda1, lambda2, chol_gram_matrix_variance)

        return - (first_term - second_term - third_term)

    def grad_dual_function(self, dual_vars, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train):
        alpha = dual_vars[:self.n]
        z = dual_vars[self.n:]

        A, f = self._compute_Ahat_and_f(dual_vars, lambda1, lambda2, chol_gram_matrix_variance, y_train)
        
        grad_wrt_alpha = -(1/(2*lambda_m)) * (gram_matrix_mean @ alpha) - (0.5 * alpha-y_train.flatten()) * f.flatten()
        grad_wrt_z = 1 - np.exp(z)*f.flatten()

        return -np.concatenate([grad_wrt_alpha, grad_wrt_z])

    def _compute_Ahat_and_f(self, dual_vars, lambda1, lambda2, chol_gram_matrix, y_train):
        alpha = dual_vars[:self.n]
        z = dual_vars[self.n:]
        inside_var = np.exp(z) + (1/4) * alpha**2 - alpha*y_train.flatten()

        A, f = Ahat_and_f(inside_var, 0, lambda1, lambda2, chol_gram_matrix)

        return A, f.reshape(-1, 1)


    def duality_gap(self, dual_vars, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train):
        """
        Compute primal objective function from dual optimization variables and resulting duality gap.
        """
        alpha = dual_vars[:self.n]
        z = dual_vars[self.n:]
        eta = (1/(2*lambda_m))*(gram_matrix_mean@alpha)

        Ahat, f = self._compute_Ahat_and_f(dual_vars, lambda1, lambda2, chol_gram_matrix_variance, y_train)

        primal = - np.sum(np.log(f.flatten()))
        primal += np.sum(((f.flatten())*(y_train.flatten()**2)))
        primal += np.sum(((eta**2) / (f.flatten())))
        primal += - 2 * np.sum(eta*y_train.flatten())
        primal += (1/(4*lambda_m)) * np.dot(alpha.T, gram_matrix_mean @ alpha)
        primal += self.lambda1 * np.trace(Ahat)
        primal += self.lambda2 * (np.linalg.norm(Ahat, "fro") ** 2)
        self.primal_value = primal
        dual = - self.dual_function(dual_vars, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train)
        self.dual_value = dual

        if dual == 0:
            return np.inf

        return np.abs((primal - dual) / dual).item()

    def check_convergence(self, dual_vars, gap_tol, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train):
        gap = self.duality_gap(dual_vars, lambda_m, lambda1, lambda2, gram_matrix_mean, chol_gram_matrix_variance, y_train)

        return (gap < gap_tol)
    
    def cv_score(self, X, y):
        """
        Negative log-likelihood score.
        """
        mean_pred, var_pred = self.predict(X=X)
        sq_residuals = (y.flatten() - mean_pred.flatten())**2

        return 0.5 * np.sum(np.log(var_pred) + (sq_residuals) / (var_pred))


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
        Train the joint heteroscedastic regression model on the given training data.

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
        self.gram_matrix_mean, _, _, self.mean_kernel, self.mean_theta, distances = make_kernel(self.X_train, self.mean_theta, self.mean_kernel, False, None, None, None)

        self.gram_matrix_variance, self.chol_gram_matrix_variance, self.chol_gram_matrix_variance_inference, self.variance_kernel, self.variance_theta, _ = make_kernel(self.X_train, self.variance_theta, self.variance_kernel, self.use_low_rank, self.method_low_rank, self.target_variance, distances)
        self.phi_gram_matrix_variance = self.chol_gram_matrix_variance

        # Solve dual optimization problem
        initial_value = np.zeros(2*self.n)

        bounds = ((None, None),) * 2 * self.n

        solver_tols = (gap_tol,)
        solver_args = (self.lambda_m, self.lambda1, self.lambda2, self.gram_matrix_mean, self.chol_gram_matrix_variance, self.y_train)

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
        self.Ahat, self.f = self._compute_Ahat_and_f(self.optimal_variables, self.lambda1, self.lambda2, self.chol_gram_matrix_variance, self.y_train)
        eta = (1 / (2 * self.lambda_m)) * (self.gram_matrix_mean.T @ self.optimal_variables[:self.n])
        self.mean = eta * self.f

    def predict(self, X: np.ndarray, chunk_size: int = 1000):
        """
        Predict mean and variance functions for input X, processed in chunks to save memory.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data where to predict the mean and variance.
        chunk_size : int
            Number of samples to process at a time.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted mean values at each input sample.
        np.ndarray of shape (n_samples,)
            Predicted variance values at each input sample.
        """
        n_samples = X.shape[0]
        results_mean = np.empty(n_samples)
        results_var = np.empty(n_samples)

        alpha = self.optimal_variables[:self.n]

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]

            X_test_chunk = self.scaler.transform(X_chunk)

            # Kernel computation
            k_train_test_chunk_mean = self.mean_kernel(self.X_train, X_test_chunk)
            eta_chunk_pred = k_train_test_chunk_mean.T @ alpha / (2 * self.lambda_m)

            precision_chunk_pred = f_pred(
                self.X_train, X_test_chunk, self.variance_kernel, self.Ahat,
                self.chol_gram_matrix_variance, self.chol_gram_matrix_variance_inference,
                self.use_low_rank, self.method_low_rank,
            )

            results_mean[start_idx:end_idx] = eta_chunk_pred / precision_chunk_pred
            results_var[start_idx:end_idx] = 1.0 / precision_chunk_pred

        return results_mean, results_var