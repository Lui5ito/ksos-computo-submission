import numpy as np

MIX_WEIGHTS = [1/2, 1/2]
MIX_MEANS   = [-0.1, 0.3]
MIX_OFFSETS = [-0.4, 0.3]
MIX_DIRECTIONS = [0.5, -0.5]


def banana(x, y, mu, y_offset, direction, b=1.0, sig_x=0.5, sig_y=0.2):
    return np.exp(-((x - mu)**2) / (2 * sig_x**2)) * \
           np.exp(-((y - y_offset - direction * b * (x - mu)**2)**2) / (2 * sig_y**2))


def sample(n, seed, pdfmax=1, n_candidates=50000):
    # Rejection sampling
    samples_x, samples_y = [], []
    rng = np.random.default_rng(seed)
    while len(samples_x) < n:
        x_cand = rng.uniform(-4, 4, n_candidates)
        y_cand = rng.uniform(-3, 3, n_candidates)
        u = rng.uniform(0, pdfmax, n_candidates)
        p_cand = MIX_WEIGHTS[0] * banana(x_cand, y_cand, MIX_MEANS[0], MIX_OFFSETS[0], direction=MIX_DIRECTIONS[0]) + \
            MIX_WEIGHTS[1] * banana(x_cand, y_cand, MIX_MEANS[1], MIX_OFFSETS[1], direction=MIX_DIRECTIONS[1])
        accept = u < p_cand
        samples_x.extend(x_cand[accept])
        samples_y.extend(y_cand[accept])

    return(np.vstack([samples_x[:n], samples_y[:n]]).T)


def sample_grid(n, sample_min, sample_max):
    sample_range = sample_max - sample_min
    xy_min = sample_min - 0.1*sample_range
    xy_max = sample_max + 0.1*sample_range
    x = np.linspace(xy_min[0], xy_max[0], n)
    y = np.linspace(xy_min[1], xy_max[1], n)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x_grid, y_grid = np.meshgrid(x, y)

    return x_grid, y_grid, dx, dy


def oracle(x, y, dx, dy):
    pdf_true_unnorm = MIX_WEIGHTS[0] * banana(x, y, MIX_MEANS[0], MIX_OFFSETS[0], direction=MIX_DIRECTIONS[0]) + \
                MIX_WEIGHTS[1] * banana(x, y, MIX_MEANS[1], MIX_OFFSETS[1], direction=MIX_DIRECTIONS[1])
    
    return (pdf_true_unnorm / (np.sum(pdf_true_unnorm) * dx * dy))
