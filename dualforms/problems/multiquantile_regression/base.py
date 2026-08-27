import numpy as np
from dualforms.cross_validation.grid_search import GridSearch
from dualforms.problems.median_regression.dual import Dual_Median
from dualforms.problems.multiquantile_regression.dual import Dual_MultiQuantile
from dualforms.problems.multiquantile_regression.dual_known_median import Dual_MultiQuantile_KnownMedian


class MultiQuantileRegression(object):
    """
    Non-crossing multi-quantile regression using kernel sum-of-squares.

    Based on the mode, solves either the joint multi-quantile regression model (see Equation (19) in the paper),
    or sequentially, the kernel quantile regression model and the multi-quantile regression model with known median (see Equation (15) in the paper).

    Parameters
    ----------
    mode : str
        Method of resolution for the multi-quantile regression. Supported values
        are ``"sequential"`` and ``"joint"``.
    quantile_levels: list
        Quantile levels to be estimated.
    median_kernel: callable
        Positive semi-definite kernel associated to the real-valued median function.
    quantile_kernels : list[callable]
        List of positive semi-definite kernels associated to each positive quantile incrementation functions.
    median_theta: object
        Lengthscales associated to median_kernel.
    quantile_thetas : list[object]
        List of lengthscales associated to quantile_kernels.
    lambda_m: float
        Regularization parameter for the ridge regularization term.
    lambda1s : list[float]
        List of regularization parameters for the trace regularization terms.
    lambda2s : list[float]
        List of regularization parameters for the Frobenius-norm regularization terms.
    """

    def __init__(
        self, 
        mode, 
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
        self.mode = mode
        self.quantile_levels = quantile_levels
        self.median_kernel = median_kernel
        self.quantile_kernels = quantile_kernels
        self.median_theta = median_theta
        self.quantile_thetas = quantile_thetas
        self.lambda_m = lambda_m
        self.lambda1s = lambda1s
        self.lambda2s = lambda2s

        self.median_model = None
        self.quantile_model = None
        self.joint_model = None
        self.solver_success = None

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
        Train the joint or sequential non-crossing multi-quantiles regression model on the given training data.

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
                self.joint_model = Dual_MultiQuantile(
                    quantile_levels=self.quantile_levels, 
                    median_kernel=self.median_kernel, quantile_kernels=self.quantile_kernels,
                    median_theta=self.median_theta, quantile_thetas=self.quantile_thetas,
                    lambda_m=self.lambda_m, lambda1s=self.lambda1s, lambda2s=self.lambda2s,
                )
                self.joint_model.train(
                    X, y, 
                    solver=solver, max_iters=max_iters, gap_tol=gap_tol, 
                    use_low_rank=use_low_rank, target_variance=target_variance, method_low_rank=method_low_rank
                )
                self.solver_success = self.joint_model.solver_success

            case "sequential":
                self.median_model = Dual_Median(median_kernel=self.median_kernel, median_theta=self.median_theta, lambda_m=self.lambda_m)
                self.median_model.train(X, y, solver=solver, max_iters=max_iters, gap_tol=gap_tol)

                self.quantile_model = Dual_MultiQuantile_KnownMedian(
                    quantile_levels=self.quantile_levels, quantile_kernels=self.quantile_kernels,
                    quantile_thetas=self.quantile_thetas, lambda1s=self.lambda1s, lambda2s=self.lambda2s
                )
                self.quantile_model.train(
                    X, y, self.median_model.median, 
                    solver=solver, max_iters=max_iters, gap_tol=gap_tol, 
                    use_low_rank=use_low_rank, target_variance=target_variance, method_low_rank=method_low_rank
                )
                self.solver_success = self.median_model.solver_success and self.quantile_model.solver_success

            case _:
                raise ValueError(f"Mode must be 'joint' or 'sequential', got {self.mode}")


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
        match self.mode:
            case "joint":
                return self.joint_model.predict(X, chunk_size)
            case "sequential":
                median_predictions = self.median_model.predict(X, chunk_size)
                return self.quantile_model.predict(X, median_predictions, chunk_size)
            case _:
                raise ValueError(f"Mode must be 'joint' or 'sequential', got {self.mode}")