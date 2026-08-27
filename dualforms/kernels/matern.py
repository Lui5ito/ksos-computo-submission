"""Code to define the Matern kernels."""

import numpy as np
import math
from scipy.spatial.distance import cdist, pdist, squareform

from dualforms.kernels import Kernel


class Matern(Kernel):
    def __init__(self, lengthscales=None, nu=2.5):
        super().__init__(lengthscales)
        self.nu = nu

    def __call__(self, X, Y=None):
        """Compute the Matérn kernel Gram matrix."""
        if self.lengthscales is None:
            lengthscales = np.ones(X.shape[1])
        else:
            lengthscales = self.lengthscales

        # Scale the data by the length scales
        X_scaled = X / lengthscales
        if Y is not None:
            Y_scaled = Y / lengthscales

        # Compute the distances
        if Y is None:
            dists = pdist(X_scaled, metric="euclidean")
        else:
            dists = cdist(X_scaled, Y_scaled, metric="euclidean")

        # Compute the kernel matrix based on the value of nu
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * math.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * math.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:  # general case; expensive to evaluate
            raise ValueError(
                "Matérn kernel is implemented only for values nu in {0.5, 1.5, 2.5, infty}."
            )

        if Y is None:
            # Convert from upper-triangular matrix to square matrix
            K = squareform(K)
            np.fill_diagonal(K, 1)

        return K