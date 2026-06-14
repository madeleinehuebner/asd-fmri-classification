from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from visualisation.common import auc_from_roc, save_figure


def metrics_from_cm(cm: np.ndarray) -> dict:
    """
    Derives ACC, SENS, SPEC, F1 from a 2x2 confusion matrix.
    Expected layout: [[TN, FP], [FN, TP]]
    """
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2 * prec * sens / (prec + sens)
        if not (np.isnan(prec) or np.isnan(sens)) and (prec + sens) > 0
        else np.nan
    )
    return {
        'acc':  (tp + tn) / total if total     > 0 else np.nan,
        'sens': sens,
        'spec': tn / (tn + fp)    if (tn + fp) > 0 else np.nan,
        'f1':   f1,
    }


def extract_metrics(all_results: dict) -> dict:
    """Converts load_folder_artefacts output to per-fold metric lists."""
    aucs, accs, senss, specs, f1s = [], [], [], [], []
    for entry in all_results.values():
        fpr = np.array(entry['fpr'], dtype=float)
        tpr = np.array(entry['tpr'], dtype=float)
        cm = np.array(entry['cm'], dtype=int)
        aucs.append(auc_from_roc(fpr, tpr))
        m = metrics_from_cm(cm)
        accs.append(m['acc'])
        senss.append(m['sens'])
        specs.append(m['spec'])
        f1s.append(m['f1'])
    return {'auc': aucs, 'acc': accs, 'sens': senss, 'spec': specs, 'f1': f1s}


def plot_metric_figure( # pragma: no cover
    experiment: str,
    cv: str,
    metric: str,
    models: list,
    dims: list,
    all_data: dict,
    model_colours: dict,
    metric_labels: dict,
    dim_labels: dict,
    model_labels: dict,
    save: bool = False,
) -> None:
    """
    One figure: three panels (feature strategies), four boxes each (models).
    """
    n_dims = len(dims)
    fig, axes = plt.subplots(
        1, n_dims,
        figsize=(4.5 * n_dims, 4),
        sharey=True,
        squeeze=False,
        gridspec_kw={'wspace': 0.06},
    )
    axes = axes[0] 

    positions = np.arange(len(models))

    for ax, dim in zip(axes, dims):
        box_data = []
        for model in models:
            result = all_data[experiment][cv][model][dim]
            if result is not None and result.get(metric):
                values = [v for v in result[metric] if not np.isnan(v)]
            else:
                values = []
            box_data.append(values)

        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.52,
            patch_artist=True,
            notch=False,
            medianprops=dict(color='#444444', linewidth=1.2),
            whiskerprops=dict(color='#444444', linewidth=1.2),
            capprops=dict(color='#444444', linewidth=1.2),
            flierprops=dict(
                marker='D',
                markerfacecolor='#999999',
                markeredgecolor='none',
                markersize=3.5,
            ),
            zorder=2,
        )

        for patch, model in zip(bp['boxes'], models):
            patch.set_facecolor(model_colours[model])
            patch.set_linewidth(1)
            patch.set_edgecolor('black')

        ax.set_xticks(positions)
        ax.set_xticklabels([model_labels[m] for m in models], fontsize=10)
        ax.set_title(dim_labels[dim], fontsize=10, pad=7)
        ax.set_xlim(-0.65, len(models) - 0.35)
        ax.tick_params(axis='y', labelsize=9)
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.02))
        ax.grid(axis='both', linestyle='--', linewidth=0.55, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel(metric_labels[metric], fontsize=11)

    # Dynamic y limits across all configurations for this metric + cv + experiment
    all_vals = [
        v
        for model in models
        for dim in dims
        for v in (all_data[experiment][cv][model][dim] or {}).get(metric, [])
        if not np.isnan(v)
    ]
    if all_vals:
        lo = max(0.0, np.percentile(all_vals,  2) - 0.06)
        hi = min(1.0, np.percentile(all_vals, 98) + 0.06)
        axes[0].set_ylim(lo, hi)

    cv_label = 'Stratified 5-Fold CV' if cv == 'SKF' else 'Leave-One-Site-Out CV'
    fig.suptitle(
        f"{metric_labels[metric]} by Model and Feature Strategy"
        f"  |  {cv_label}  |  {experiment}",
        fontsize=12, y=1.02,
    )

    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.15)
    save_figure(fig, f"{metric}_{cv.lower()}_{experiment.lower()}_boxplot.pdf", save)
