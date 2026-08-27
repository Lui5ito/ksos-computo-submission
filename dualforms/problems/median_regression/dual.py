import numpy as np
from dualforms.problems.utils.utils_solver import solve_dispatch
from dualforms.problems.utils.utils_kernels import make_kernel
from sklearn.preprocessing import StandardScaler

class Dual_Median(object):
    """
    Kernel quantile regression with the 0.5-pinball loss.

    This class implements kernel quantile regression, 
    where the problem is formulated and solved through its dual 
    optimization problem.

    Parameters
    ----------
    median_kernel : callable
        Positive semi-definite kernel associated to the real-valued median function.
    median_theta : object
        Lengthscales associated to median_kernel.
    lambda_m : float
        Regularization parameter for the ridge regularization term.
    """

    def __init__(
        self,
        median_kernel,
        median_theta,
        lambda_m: float
    ):

        # Kernel quantile regression problem definition
        self.median_kernel = median_kernel
        self.median_theta = median_theta
        self.lambda_m = lambda_m

        # Training sample variables
        self.n = None
        self.d = None
        self.X_train = None
        self.y_train = None

        # Optimization variables
        self.gammahat = None
        self.eta = None
        self.optimal_variables = None

        # Solver variables
        self.solver_state = None
        self.solver_iter = None
        self.solver_time = None
        self.solver_success = None

    def dual_function(self, dual_vars, lambda_m, gram_matrix_median, y_train):
        first_term = np.dot(y_train.flatten(), dual_vars)
        second_term = (1 / (4 * lambda_m)) * np.dot(dual_vars, gram_matrix_median @ dual_vars)

        return -(first_term - second_term)

    def grad_dual_function(self, dual_vars, lambda_m, gram_matrix_median, y_train):
        y = y_train.flatten()
        grad = y - (1 / (2 * lambda_m)) * (gram_matrix_median @ dual_vars)

        return -grad

    def duality_gap(self, dual_vars, lambda_m, gram_matrix_median, y_train):
        """
        Compute primal objective function from dual optimization variables and resulting duality gap.
        """
        median = (1 / (2 * lambda_m)) * (gram_matrix_median @ dual_vars)
        y = y_train.flatten()

        primal = 0.5 * np.sum(np.abs(y - median))
        primal += (1 / (4 * lambda_m )) * np.dot(dual_vars, gram_matrix_median @ dual_vars)
        self.primal_value = primal

        dual = -self.dual_function(dual_vars, lambda_m, gram_matrix_median, y_train)
        self.dual_value = dual
        if dual == 0:
            return np.inf

        return np.abs((primal - dual) / dual).item()

    def check_convergence(self, dual_vars, gap_tol, lambda_m, gram_matrix_median, y_train):
        gap = self.duality_gap(dual_vars, lambda_m, gram_matrix_median, y_train)

        return gap < gap_tol

    def cv_score(self, X, y):
        """
        Sum of median pinball loss score.
        """
        predictions = self.predict(X=X)
        loss = np.sum(np.abs(y.flatten() - predictions.flatten()))
        
        return loss

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        solver,
        max_iters: int,
        gap_tol: float,
    ):
        """
        Train the kernel quantile regression model on the given training data.

        The input data are standardized before fitting. The method constructs
        the kernel matrices, solves the dual optimization problem, 
        checks the convergence of the dual to the primal
        and stores the resulting optimal parameters in the instance.

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

        # Assemble kernel matrices on training sample
        self.gram_matrix_median, _, _, self.median_kernel, self.median_theta, _ = make_kernel(self.X_train, self.median_theta, self.median_kernel, False, None, None, None)

        # Solve dual optimization problem
        initial_value = np.zeros(self.n)

        bounds = ((-0.5, 0.5),) * self.n

        solver_tols = (gap_tol,)
        solver_args = (self.lambda_m, self.gram_matrix_median, self.y_train)

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
        self.median = (1 / (2 * self.lambda_m)) * (self.gram_matrix_median.T @ self.optimal_variables)

        # Convergence check
        self.solver_success = res.solver_success
        self.solver_iter = res.nit
        self.solver_state = res.message
        tag = "[OPTIM: CONVERGED]" if self.solver_success else "[OPTIM: FAILED]"
        print(f"{tag} Solver={solver} | Iterations={self.solver_iter}")


    def predict(self, X: np.ndarray, chunk_size: int = 1000):
        """
        Predict median function for input X, processed in chunks to save memory.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data where to predict the median.
        chunk_size : int
            Number of samples to process at a time.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted median values at each input sample.
        """
        n_samples = X.shape[0]
        results = np.empty(n_samples)

        alpha = self.optimal_variables

        for start_idx in range(0, n_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, n_samples)
            X_chunk = X[start_idx:end_idx]

            X_test_chunk = self.scaler.transform(X_chunk)

            k_train_test_chunk_median = self.median_kernel(self.X_train, X_test_chunk)
            results[start_idx:end_idx] = (1 / (2 * self.lambda_m)) * (k_train_test_chunk_median.T @ alpha)

        return results
