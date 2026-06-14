"""Fixtures for src/visualisation tests."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def patch_display_path(monkeypatch):
    monkeypatch.setattr("visualisation.common.display_path", lambda p: p)


@pytest.fixture()
def fake_artefacts():
    """
    Fake load_folder_artefacts output: 3 folds, each with fpr, tpr, cm arrays.
    Uses a near-perfect classifier so all metrics are in [0, 1].
    """
    rng = np.random.default_rng(0)
    folds = {}
    for i in range(3):
        fpr = np.array([0.0, 0.05, 0.1, 1.0])
        tpr = np.array([0.0, 0.85, 0.95, 1.0])
        cm = np.array([[45, 5], [3, 47]])
        folds[f"fold_{i}.json"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "cm": cm.tolist()}
    return folds
