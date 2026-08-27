import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.cm as cm

from .common import SCATTER_TRAIN

def get_levels_and_labels(pdf, quantiles, dx, dy):
    pdf_flat_sorted = np.sort(pdf.flatten())[::-1]
    cumulative_mass = np.cumsum(pdf_flat_sorted) * dx * dy
    
    levels = []
    labels = {}
    
    for q in quantiles:
        idx = np.searchsorted(cumulative_mass, q)
        threshold = pdf_flat_sorted[idx] if idx < len(pdf_flat_sorted) else pdf_flat_sorted[-1]
        levels.append(threshold)
        labels[threshold] = q
        
    levels_sorted = sorted(levels)
    return levels_sorted, labels

def plot_density_2D(X_train, x_grid, y_grid, dx, dy, density_pred, true_density, *, title="", integral=None, save_path=None):
    quantiles_plot = [0.1, 0.2, 0.5, 0.8, 0.9, 0.95]
    levels_true, labels_true = get_levels_and_labels(true_density, quantiles_plot, dx, dy)
    levels_pred, _ = get_levels_and_labels(density_pred, quantiles_plot, dx, dy)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(X_train[:, 0], X_train[:, 1], **SCATTER_TRAIN)

    cmap = cm.viridis
    line_colors = [cmap(q) for q in quantiles_plot]
    contour_true = ax.contour(x_grid, y_grid, true_density, levels=levels_true, colors=line_colors, linestyles='--', linewidths=1, zorder=2)
    ax.clabel(contour_true, contour_true.levels, inline=True, fmt=labels_true, fontsize=11)
    ax.contour(x_grid, y_grid, density_pred, levels=levels_pred, colors=line_colors, linewidths=2, zorder=2)

    legend_handles = [
        Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='True density'),
        Line2D([0], [0], color='black', linestyle='-', linewidth=2, label='Predicted density'),
        Line2D([0], [0], marker='o', color='none', markerfacecolor=SCATTER_TRAIN["c"], alpha=SCATTER_TRAIN["alpha"], markersize=6, label='Training data'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', frameon=True, framealpha=0.9)

    if integral is not None:
        box_text = f"Integral = {integral:.3f}"
        ax.text(
            0.4, 0.95, box_text,
            transform=ax.transAxes,
            fontsize=14, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.3", alpha=0.85),
        )

    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    ax.grid(True, linestyle=':', alpha=0.6)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
    return fig
