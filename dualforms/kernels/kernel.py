from abc import ABCMeta
import numpy as np


class Kernel(metaclass=ABCMeta):
    """
    Base class for all kernels.
    """

    def __init__(self, lengthscales=None):
        self.lengthscales = self._process_lengthscales(lengthscales)

    def _process_lengthscales(self, lengthscales):
        if lengthscales is None:
            return None
        lengthscales = np.asarray(lengthscales)
        if lengthscales.ndim == 2 and lengthscales.shape[1] == 1:
            lengthscales = lengthscales.flatten()
        return lengthscales

    def update(self, new_lengthscales):
        self.lengthscales = self._process_lengthscales(new_lengthscales)

    def __call__(self, X, Y=None):
        raise NotImplementedError(
            "The __call__ method should be implemented by subclasses."
        )
