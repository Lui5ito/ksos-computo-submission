"""Code to harmonise the call to sample data for each cases."""

import numpy as np

from .unsupervised_cases import (
    banana,
)  # Import case-specific logic
from .data import Data


class Unsupervised(Data):

    def __init__(
        self,
        name: str,
        input_dim: int = 2,
        output_dim: int = 1,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(name, verbose)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.case_kwargs = kwargs

        self.case_map = {
            "banana": banana,
        }

    def sample(self, n, seed):
        """
        Generate Monte-Carlo sample (x_i) of size n for a given seed.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.sample(n, seed, **self.case_kwargs)

    def sample_grid(self, n, sample_min, sample_max):
        """
        Generate regular grid of equally-spaced values in the feature space of size n.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.sample_grid(n, sample_min, sample_max, **self.case_kwargs)

    def oracle(self, x, y, dx, dy):
        """
        Compute oracle probability density function at features (x,y) given by a regular grid from sample_grid.
        Normalization constant is estimated using the grid spacings dx and dy.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.oracle(x, y, dx, dy, **self.case_kwargs)
