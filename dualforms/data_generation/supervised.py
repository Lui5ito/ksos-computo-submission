"""Code to harmonise the call to sample data for each cases."""

import numpy as np

from .supervised_cases import (
    case_1,
    case_2,
)  # Import case-specific logic
from .data import Data


class Supervised(Data):

    def __init__(
        self,
        name: str,
        input_dim: int,
        output_dim: int,
        verbose: int = 0,
        **kwargs,
    ):
        super().__init__(name, verbose)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.case_kwargs = kwargs

        self.case_map = {
            "case_1": case_1,
            "case_2": case_2,
        }

    def sample(self, n, seed):
        """
        Generate Monte-Carlo sample (x_i,y_i) of size n for a given seed.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.sample((n, self.input_dim), seed, **self.case_kwargs)

    def sample_grid(self, n, seed):
        """
        Generate Monte-Carlo sample y_i for equally-spaced values x_i of features of size n for a given seed.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.sample_grid((n, self.input_dim), seed, **self.case_kwargs)

    @property
    def ylim(self):
        case = self.case_map.get(self.name)
        return getattr(case, "YLIM", None)

    def oracle(self, X, alpha=0.1):
        """
        Compute oracle mean, standard deviation and confidence intervals at level alpha.
        """
        case = self.case_map.get(self.name)
        if not case:
            raise ValueError(f'Unsupported case: {self.name}')
        return case.oracle(X, alpha, **self.case_kwargs)
