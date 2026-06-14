from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from data_io.centralised_logger import CentralisedLogger
from utils.paths import display_path
from visualisation.common import auc_from_roc, discover_timestamps, save_figure


def interpolate_roc(fpr: np.ndarray, tpr: np.ndarray, n: int = 200) -> tuple:
    """Interpolates TPR onto a fixed FPR grid for averaging across folds or sites."""
    fpr_grid = np.linspace(0, 1, n)
    tpr_interp = np.interp(fpr_grid, np.sort(fpr), tpr[np.argsort(fpr)])
    return fpr_grid, tpr_interp


def site_label(filename: str) -> str:
    """Extracts the site name from an artefact filename."""
    clean = filename.split()[0].split('_')[0]
    if clean.lower() == 'site' and '_' in filename:
        clean = filename.split('_')[1]
    return clean


def _style_ax(ax) -> None:
    ax.set_axisbelow(True)
    ax.grid(linestyle='--', linewidth=0.5, alpha=0.35, zorder=0)


def load_raw(model: str, dim: str, cv: str, exp: str, runs: dict) -> dict | None:
    """Loads artefacts for a given (model, dim, cv, experiment) combination."""
    key = (model, dim)
    if key not in runs:
        return None
    folder_name, tags = runs[key]
    target_tag = f'{cv}-{exp}'

    ts_index = next((i for i, t in tags.items() if t == target_tag), None)
    if ts_index is None:
        return None

    timestamps = discover_timestamps(folder_name)
    if ts_index >= len(timestamps):
        print(f"WARNING: index {ts_index} out of range for {folder_name} "
              f"({len(timestamps)} runs found). skipping")
        return None

    logger = CentralisedLogger(model_name=folder_name)
    target_folder = logger.artefact_path / timestamps[ts_index]
    if not target_folder.exists():
        print(f"WARNING: folder not found {target_folder}")
        return None

    return logger.load_folder_artefacts(folder_path=target_folder)


def plot_skf_roc( # pragma: no cover
    dim: str,
    exp: str,
    models: list,
    runs: dict,
    model_colours: dict,
    dim_labels: dict,
    save: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=300)
    legend_lines = []

    for model in models:
        colour = model_colours[model]
        raw = load_raw(model, dim, 'SKF', exp, runs)
        if raw is None:
            continue

        tpr_matrix = []
        for entry in raw.values():
            fpr = np.array(entry['fpr'], dtype=float)
            tpr = np.array(entry['tpr'], dtype=float)
            fpr_g, tpr_i = interpolate_roc(fpr, tpr)
            tpr_matrix.append(tpr_i)
            # ax.plot(fpr_g, tpr_i, color=colour, linewidth=0.6, alpha=0.22, zorder=2)

        if tpr_matrix:
            mean_tpr = np.mean(tpr_matrix, axis=0)
            mean_auc = auc_from_roc(fpr_g, mean_tpr)
            ax.plot(fpr_g, mean_tpr, color=colour, linewidth=1, alpha=0.9, zorder=3)
            legend_lines.append(
                Line2D([0], [0], color=colour, linewidth=1,
                       label=f"{model}  (AUC = {mean_auc:.3f})")
            )

    ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=0.9, linestyle=':', zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(
        f'ROC Curves  |  Stratified 5-Fold CV  |  {exp}\n{dim_labels[dim]}',
        fontsize=11, pad=8,
    )
    ax.legend(handles=legend_lines, frameon=True, fontsize=9, loc='lower right')
    _style_ax(ax)
    save_figure(fig, f"roc_loso_{dim.lower()}_{model.lower()}_{exp.lower()}.pdf", save)


def plot_loso_roc( # pragma: no cover
    dim: str,
    model: str,
    exp: str,
    runs: dict,
    model_colours: dict,
    dim_labels: dict,
    save: bool = False,
) -> None:
    colour = model_colours[model]
    raw = load_raw(model, dim, 'LOSO', exp, runs)
    if raw is None:
        print(f'No LOSO data for {model} / {dim} / {exp}. skipping.')
        return

    fig, ax = plt.subplots(figsize=(6.0, 5.2), dpi=300)
    tpr_matrix = []
    legend_lines = []

    n_sites = len(raw)
    site_cmap = matplotlib.colormaps.get_cmap('tab20')
    site_colours = [site_cmap(i / max(n_sites - 1, 1)) for i in range(n_sites)]

    for i, (fname, entry) in enumerate(raw.items()):
        fpr = np.array(entry['fpr'], dtype=float)
        tpr = np.array(entry['tpr'], dtype=float)
        fpr_g, tpr_i = interpolate_roc(fpr, tpr)
        tpr_matrix.append(tpr_i)
        site_auc = auc_from_roc(fpr, tpr)
        site = site_label(fname)

        ax.plot(fpr_g, tpr_i, color=site_colours[i], linewidth=1.2, alpha=0.6, zorder=2)
        legend_lines.append(
            Line2D([0], [0], color=site_colours[i], linewidth=1.5,
                   label=f"{site}  ({site_auc:.2f})")
        )

    if tpr_matrix:
        mean_tpr = np.mean(tpr_matrix, axis=0)
        mean_auc = auc_from_roc(fpr_g, mean_tpr)
        ax.plot(fpr_g, mean_tpr, color='black', linewidth=1.5, alpha=1, zorder=4)
        legend_lines.append(
            Line2D([0], [0], color='black', linewidth=1.5,
                   label=f"Mean  (AUC = {mean_auc:.3f})")
        )

    ax.plot([0, 1], [0, 1], color='#bbbbbb', linewidth=0.9, linestyle=':', zorder=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.set_title(
        f'ROC Curves  |  LOSO  |  {model}  |  {exp}\n{dim_labels[dim]}',
        fontsize=9, pad=8,
    )
    ax.legend(
        handles=legend_lines,
        frameon=True,
        fontsize=7.5,
        loc='lower right',
        borderaxespad=1.0,
    )
    _style_ax(ax)
    save_figure(fig, f"roc_loso_{dim.lower()}_{model.lower()}_{exp.lower()}.pdf", save)

