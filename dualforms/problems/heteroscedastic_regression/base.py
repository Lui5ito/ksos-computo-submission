import numpy as np
from dualforms.cross_validation.grid_search import GridSearch
from dualforms.problems.mean_regression.dual import Dual_Mean
from dualforms.problems.heteroscedastic_regression.dual import Dual_Heteroscedastic
from dualforms.problems.heteroscedastic_regression.dual_known_mean import Dual_Heteroscedastic_KnownMean


class HeteroscedasticRegression(object):
    """
    Heteroscedastic regression using kernel sum-of-squares.

    Based on the mode, solves either the joint heteroscedastic regression model (see Equation (18) in the paper),
    or sequentially, the kernel ridge regression model and the heteroscedastic regression model with known mean (see Equation (14) in the paper).

    Parameters
    ----------
    mode : str
        Method of resolution for the heteroscedastic regression. Supported values
        are ``"sequential"`` and ``"joint"``.
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
        mode, 
        mean_kernel,
        variance_kernel,
        mean_theta,
        variance_theta,
        lambda_m: float,
        lambda1: float,
        lambda2: float,
    ):

        # kSoS problem definition
        self.mode = mode
        self.mean_kernel = mean_kernel
        self.variance_kernel = variance_kernel
        self.mean_theta = mean_theta
        self.variance_theta = variance_theta
        self.lambda_m = lambda_m
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.mean_model = None
        self.variance_model = None
        self.joint_model = None
        self.solver_success = None

    def train(
        self, 
        X, 
        y, 
        solver,
        max_iters: int,
        gap_tol: float,
        use_low_rank: bool,
        target_variance: float,
        method_low_rank: str = "svd"
    ):
        """
        Train the joint or sequential heteroscedastic regression model on the given training data.

        Based on the mode, use the corresponding train method.

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

        match self.mode:
            case "joint":
                self.joint_model = Dual_Heteroscedastic(
                    mean_kernel=self.mean_kernel, variance_kernel=self.variance_kernel,
                    mean_theta=self.mean_theta, variance_theta=self.variance_theta,
                    lambda_m=self.lambda_m, lambda1=self.lambda1, lambda2=self.lambda2,
                )
                self.joint_model.train(
                    X, y, 
                    solver=solver, max_iters=max_iters, gap_tol=gap_tol, 
                    use_low_rank=use_low_rank, target_variance=target_variance, method_low_rank=method_low_rank
                )
                self.solver_success = self.joint_model.solver_success

            case "sequential":
                self.mean_model = Dual_Mean(mean_kernel=self.mean_kernel, mean_theta=self.mean_theta, lambda_m=self.lambda_m)
                self.mean_model.train(X, y, solver=solver, max_iters=max_iters, gap_tol=gap_tol)

                self.variance_model = Dual_Heteroscedastic_KnownMean(
                    variance_kernel=self.variance_kernel, variance_theta=self.variance_theta,
                    lambda1=self.lambda1, lambda2=self.lambda2,
                )
                self.variance_model.train(
                    X, y, self.mean_model.mean, 
                    solver=solver, max_iters=max_iters, gap_tol=gap_tol, 
                    use_low_rank=use_low_rank, target_variance=target_variance, method_low_rank=method_low_rank
                )
                self.solver_success = self.mean_model.solver_success and self.variance_model.solver_success

            case _:
                raise ValueError(f"Mode must be 'joint' or 'sequential', got {self.mode}")


    def predict(self, X: np.ndarray, chunk_size: int = 1000):
        """
       Based on the mode, use the corresponding predict method.

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
        match self.mode:
            case "joint":
                return self.joint_model.predict(X, chunk_size)
            case "sequential":
                return self.mean_model.predict(X, chunk_size), self.variance_model.predict(X, chunk_size)
            case _:
                raise ValueError(f"Mode must be 'joint' or 'sequential', got {self.mode}")