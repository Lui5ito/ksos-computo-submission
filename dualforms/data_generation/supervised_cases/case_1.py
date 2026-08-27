"""Test case 1, inspired by Adaptive design and analysis of supercomputer experiments by Robert B. Gramacy and Herbert K. H. Lee, see section 4.1. Complex mean, heterosedastic variance."""

import numpy as np
import scipy.stats as stats

YLIM = (-3, 5)


def mean(x):
    x = 10 * (1 + x)
    result = np.where(
        x <= 9.6,
        np.sin(np.pi * x / 5) + np.cos(4 * np.pi * x / 5) / 5,  # When x <= 9.6
        -1 + x / 10,  # When x > 9.6
    )

    return np.sum(result, axis=1).reshape(-1, 1)


def std_dev(x):
    return np.sqrt(0.1 + 2 * (x**2))


def sample(input_shape, seed):
    X = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=input_shape)
    epsilon = np.random.default_rng(seed=seed).normal(
        loc=0, scale=1, size=(input_shape[0], 1)
    )
    y = mean(X) + std_dev(X) * epsilon

    return X, y


def sample_grid(input_shape, seed):
    X = np.linspace(-1, 1, input_shape[0]).reshape(input_shape)
    epsilon = np.random.default_rng(seed=seed).normal(
        loc=0, scale=1, size=(input_shape[0], 1)
    )
    y = mean(X) + std_dev(X) * epsilon

    return X, y


def oracle(X, alpha):
    mean_vals = mean(X)
    std_vals = std_dev(X)
    lower_bound = mean_vals - std_vals * stats.norm.ppf(1 - alpha / 2)
    upper_bound = mean_vals + std_vals * stats.norm.ppf(1 - alpha / 2)
    
    return mean_vals, std_vals, lower_bound, upper_bound
