import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from .common import SCATTER_TRAIN, SCATTER_TEST


def plot_heteroscedastic(
    X_train, y_train, X_test, y_test,
    mean_pred, var_pred,
    true_mean=None, true_lo=None, true_hi=None,
    *,
    alpha=0.05,
    title="",
    rmse_mean=None,
    rmse_var=None,
    save_path=None,
):
    std_pred = np.sqrt(var_pred)
    z = norm.ppf(1 - alpha / 2)
    sort_idx = np.argsort(X_test.ravel())
    x  = X_test.ravel()[sort_idx]
    mu = mean_pred[sort_idx]
    lo = (mean_pred - z * std_pred)[sort_idx]
    hi = (mean_pred + z * std_pred)[sort_idx]

    fig, ax = plt.subplots(figsize=(10, 6))

    if true_mean is not None and true_lo is not None and true_hi is not None:
        ax.plot(x, true_lo.ravel()[sort_idx],
                color="orange", lw=1.0, ls="--", alpha=0.5, zorder=2, label=f"True {1-alpha:.0%} band")
        ax.plot(x, true_hi.ravel()[sort_idx],
                color="orange", lw=1.0, ls="--", alpha=0.5, zorder=2)
        ax.plot(x, true_mean.ravel()[sort_idx],
                color="red", lw=1.0, ls="--", alpha=0.5, zorder=2, label="True mean")

    ax.plot(x, lo, color="orange", lw=1.5, alpha=0.9, zorder=3, label=f"Predicted {1-alpha:.0%} band")
    ax.plot(x, hi, color="orange", lw=1.5, alpha=0.9, zorder=3)
    ax.plot(x, mu, color="red",   lw=1.5, alpha=0.9, zorder=3, label="Predicted mean")

    ax.scatter(X_train.ravel(), y_train.ravel(), **SCATTER_TRAIN, label="Training data")
    if y_test is not None:
        ax.scatter(X_test.ravel(), y_test.ravel(), **SCATTER_TEST, label="test")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left")
    ax.autoscale_view()
    x_all = np.concatenate([x, X_train.ravel()])
    ax.set_xlim(x_all.min(), x_all.max())
    if title:
        ax.set_title(title)
    if rmse_mean is not None or rmse_var is not None:
        box_text = f"RMSE mean = {rmse_mean:.3e}\nRMSE variance = {rmse_var:.3e}"
        ax.text(
            0.65, 0.94, box_text,
            transform=ax.transAxes,
            fontsize=14, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.85),
        )
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
