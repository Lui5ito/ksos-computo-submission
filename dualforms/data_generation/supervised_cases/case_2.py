import numpy as np
import scipy.stats as stats

YLIM = None


def mean(x):
    return np.sin(x)


def _get_sigmas(x):
    sigma_lower = 0.5

    mu = np.sin(x)
    sigma_upper = 0.4 * (mu + 1) + 0.1

    return sigma_lower, sigma_upper


def std_dev(x):
    _, sigma_upper = _get_sigmas(x)

    return sigma_upper


def sample(input_shape, seed):
    rng = np.random.default_rng(seed=seed)

    X = rng.uniform(low=0, high=4 * np.pi, size=input_shape)
    z = rng.standard_normal(size=(input_shape[0], 1))

    sigma_lower, sigma_upper = _get_sigmas(X)

    noise = np.where(
        z < 0,
        z * sigma_lower,
        z * sigma_upper
    )

    y = mean(X) + noise

    return X, y


def sample_grid(input_shape, seed):
    X = np.linspace(0, 4 * np.pi, input_shape[0]).reshape(input_shape)
    rng = np.random.default_rng(seed=seed)
    z = rng.standard_normal(size=(input_shape[0], 1))

    sigma_lower, sigma_upper = _get_sigmas(X)

    noise = np.where(
        z < 0,
        z * sigma_lower,
        z * sigma_upper
    )

    y = mean(X) + noise

    return X, y


def oracle(X, alpha):
    mean_vals = mean(X)
    var_vals = _get_sigmas(X)

    z_crit = stats.norm.ppf(1 - alpha / 2)

    lower_bound = mean_vals - z_crit * var_vals[0]
    upper_bound = mean_vals + z_crit * var_vals[1]

    return mean_vals, var_vals, lower_bound, upper_bound
