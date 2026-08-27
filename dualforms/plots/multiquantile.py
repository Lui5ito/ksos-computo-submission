import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .common import SCATTER_TEST, SCATTER_TRAIN


def compute_true_quantiles(dataset, X_test, quantile_levels):
    tqs = {0.5: dataset.oracle(X_test, alpha=0.99)[0]}
    for tau in quantile_levels:
        alpha = 2 * min(tau, 1 - tau)
        _, _, lo, up = dataset.oracle(X_test, alpha=alpha)
        tqs[tau] = lo if tau < 0.5 else up
    return tqs


def _tau_colors(all_taus):
    cmap = cm.viridis
    return dict(zip(all_taus, cmap(np.linspace(0, 1, len(all_taus)))))


def draw_quantile_bands(
    ax, x, sort_idx, preds, tqs, X_train, y_train,
    lower_taus, upper_taus, title=None, title_fontsize=9,
    X_test=None, y_test=None,
):
    all_taus = sorted(lower_taus + [0.5] + upper_taus)
    color_map = _tau_colors(all_taus)
    x_end = x[-1]

    for tau in all_taus:
        color = color_map[tau]
        pred_sorted = preds[tau][sort_idx]
        true_sorted = tqs[tau][sort_idx]

        ax.plot(x, true_sorted, color=color, lw=1.0, ls="--", alpha=0.5, zorder=2)
        ax.plot(x, pred_sorted, color=color, lw=1.5, alpha=0.9, zorder=3)

        ax.annotate(
            f"{tau:.1f}",
            xy=(x_end, pred_sorted[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            fontsize=7,
            color=color,
        )

    ax.scatter(X_train.ravel(), y_train.ravel(), **SCATTER_TRAIN)
    if X_test is not None and y_test is not None:
        ax.scatter(X_test.ravel(), y_test.ravel(), **SCATTER_TEST)

    span = x_end - x[0]
    ax.set_xlim(x[0], x_end + 0.06 * span)

    if title is not None:
        ax.set_title(title, fontsize=title_fontsize)


def plot_multiquantile_comparison(
    X_train, y_train, X_test, y_test,
    predictions_by_row, true_quantiles, quantile_levels,
    row_labels, rmse_by_row,
    zoom_left, zoom_right,
    *,
    rmse_median_by_row=None,
    save_path=None,
):
    sort_idx = np.argsort(X_test.ravel())
    x = X_test.ravel()[sort_idx]
    all_taus   = sorted(quantile_levels)
    lower_taus = [t for t in all_taus if t < 0.5]
    upper_taus = [t for t in all_taus if t > 0.5]

    n_rows = len(row_labels)
    fig, axes = plt.subplots(
        n_rows, 3, figsize=(14, 4.5 * n_rows),
        gridspec_kw={"width_ratios": [1, 2, 1]},
    )
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    xlo_l, xhi_l, ylo_l, yhi_l = zoom_left
    xlo_r, xhi_r, ylo_r, yhi_r = zoom_right

    rmse_median_by_row = rmse_median_by_row or [None] * n_rows

    for row, (label, preds, rmse, rmse_median) in enumerate(
        zip(row_labels, predictions_by_row, rmse_by_row, rmse_median_by_row)
    ):
        for col in range(3):
            draw_quantile_bands(
                axes[row, col], x, sort_idx, preds, true_quantiles,
                X_train, y_train, lower_taus, upper_taus,
                X_test=None, y_test=None,
            )

        axes[row, 0].set_xlim(xlo_l, xhi_l)
        axes[row, 0].set_ylim(ylo_l, yhi_l)
        axes[row, 2].set_xlim(xlo_r, xhi_r)
        axes[row, 2].set_ylim(ylo_r, yhi_r)

        for (xmin, xmax, ymin, ymax), zoom_label in [(zoom_left, "zoom left"), (zoom_right, "zoom right")]:
            axes[row, 1].add_patch(Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                facecolor="1", edgecolor="0.5", lw=0.8, alpha=0.5, zorder=1,
            ))

        legend_handles = [
            Line2D([0], [0], color="0.3", ls="--", lw=1.0, alpha=0.5, label="True quantile"),
            Line2D([0], [0], color="0.3", lw=1.5, alpha=0.9, label="Predicted quantile"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=SCATTER_TRAIN["c"],
                   alpha=SCATTER_TRAIN["alpha"], markersize=6, label="Training data"),
        ]
        axes[row, 1].legend(handles=legend_handles, loc="lower left", fontsize=8, frameon=True, framealpha=0.9)

        box_text = f"RMSE aggregated = {rmse:.4f}"
        if rmse_median is not None:
            box_text += f"\nRMSE median = {rmse_median:.4f}"
        axes[row, 1].text(
            0.8, 0.95, box_text,
            transform=axes[row, 1].transAxes,
            fontsize=14, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.85),
        )
        axes[row, 0].set_ylabel(f"{label}\ny", fontsize=12)

    for col, t in enumerate(["Zoom (left)", "Full", "Zoom (right)"]):
        axes[0, col].set_title(t, fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("x", fontsize=9)

    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig

