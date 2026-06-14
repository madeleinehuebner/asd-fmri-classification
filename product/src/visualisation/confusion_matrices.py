from __future__ import annotations
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib.colors import LinearSegmentedColormap

from visualisation.common import save_figure

CV_DISPLAY = {
    'SKF':  'Stratified 5-Fold CV',
    'LOSO': 'Leave-One-Site-Out CV',
}


def default_title(label: str, cv: str, exp: str) -> str:
    return f"{label}\n{CV_DISPLAY[cv]}  |  {exp}"


def default_filename(label: str, cv: str, exp: str) -> str:
    slug = label.lower().replace(' ', '_').replace('+', '').replace('__', '_')
    return f"cm_{slug}_{cv.lower()}_{exp.lower()}.pdf"


def plot_confusion_matrix( # pragma: no cover
    cm: np.ndarray,
    title: str,
    class_labels: list,
    save: bool = False,
) -> None:
    """
    Plots a row-normalised confusion matrix with raw counts annotated.
    Layout: [[TN, FP], [FN, TP]] displayed as [[TD, ASD], [TD, ASD]].
    """
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm / row_sums, 0.0)

    cmap = LinearSegmentedColormap.from_list(
        'cm_blue', ['#ffffff', 'palevioletred']
    )

    fig, ax = plt.subplots(figsize=(4.2, 3.8), dpi=300)
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=cmap, vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Proportion', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for i in range(2):
        for j in range(2):
            prop = cm_norm[i, j]
            count = cm[i, j]
            rect = patches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                fill=False, edgecolor='black', linewidth=1
            )
            ax.add_patch(rect)
            ax.text(
                j, i,
                f'{prop:.2f}\n({count})',
                ha='center', va='center',
                fontsize=10
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_labels, fontsize=10)
    ax.set_yticklabels(class_labels, fontsize=10)
    ax.set_xlabel('Predicted class', fontsize=10, labelpad=8)
    ax.set_ylabel('True class', fontsize=10, labelpad=8)
    ax.set_title(title, fontsize=10, pad=10)

    fig.subplots_adjust(left=0.15, right=0.9, top=0.88, bottom=0.15)
    slug = re.sub(r'[^\w]+', '_', title.replace('\n', '_').lower()).strip('_')
    save_figure(fig, f"cm_{slug}.pdf", save)
