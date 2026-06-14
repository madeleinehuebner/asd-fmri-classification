from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data_io.centralised_logger import CentralisedLogger
from utils.paths import display_path, get_project_root

def figures_dir() -> Path:
    """Returns product/notebooks/visualisation/figures/, creating it if needed."""
    root = get_project_root()
    path = root / 'product' / 'notebooks' / 'visualisation' / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, filename: str, save: bool) -> None:
    """Saves fig to figures_dir/filename and closes it. No-op when save is False."""
    if not save:
        return
    path = figures_dir() / filename
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {display_path(path)}')


def discover_timestamps(folder_name: str) -> list[str]:
    """Returns all timestamp subfolders for a given model folder, sorted oldest first."""
    logger = CentralisedLogger(model_name=folder_name)
    artefact_path = logger.artefact_path
    if not artefact_path.exists():
        return []
    return sorted(
        p.name for p in artefact_path.iterdir()
        if p.is_dir() and not p.name.startswith('.')
    )


def auc_from_roc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """AUC via trapezoidal rule; handles unsorted fpr."""
    order = np.argsort(fpr)
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz'))
    return float(_trapz(tpr[order], fpr[order]))
