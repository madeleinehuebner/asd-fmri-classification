"""
Fixtures for src/data_io tests.

Tier-2 fixtures: scoped to the data_io test directory.
"""

import pytest
import numpy as np
import pandas as pd
import torch


@pytest.fixture(autouse=True)
def patch_display_path(monkeypatch):
    """
    display_path() calls get_project_root(), which fails outside the real project
    tree (e.g. in tmp_path). It is only used in print statements, so replacing it
    with an identity function keeps tests focused on functional behaviour.
    """
    identity = lambda p: p  # noqa: E731
    monkeypatch.setattr("data_io.save_load_dataset.display_path", identity)
    monkeypatch.setattr("data_io.centralised_logger.display_path", identity)


# Used in: test_save_load_dataset.py
@pytest.fixture()
def sample_1d_dir(tmp_path):
    """Temp folder containing three .1D timeseries files and one unrelated file."""
    np.random.seed(0)
    for i in range(3):
        f = tmp_path / f"subject_{i:04d}_rois_cc200.1D"
        np.savetxt(str(f), np.random.randn(50, 10))
    (tmp_path / "readme.txt").write_text("not a timeseries file")
    return tmp_path


# Used in: test_save_load_dataset.py
@pytest.fixture()
def sample_pheno_csv_path(tmp_path):
    """Temp CSV with all 20 default ABIDE phenotypic columns."""
    df = pd.DataFrame({
        'subject':               [51001, 51002, 51003],
        'SITE_ID':               ['NYU', 'NYU', 'YALE'],
        'DX_GROUP':              [1, 2, 1],
        'AGE_AT_SCAN':           [10.5, 12.3, 15.0],
        'SEX':                   [1, 2, 1],
        'FIQ':                   [100, 110, 95],
        'VIQ':                   [102, 108, 97],
        'PIQ':                   [98, 112, 93],
        'HANDEDNESS_CATEGORY':   ['R', 'R', 'L'],
        'EYE_STATUS_AT_SCAN':    [1, 1, 2],
        'func_mean_fd':          [0.10, 0.20, 0.15],
        'func_num_fd':           [5, 10, 7],
        'func_perc_fd':          [0.05, 0.10, 0.07],
        'func_dvars':            [0.80, 0.90, 0.85],
        'func_outlier':          [0.00, 0.10, 0.05],
        'func_efc':              [0.50, 0.60, 0.55],
        'func_fber':             [100.0, 110.0, 105.0],
        'func_fwhm':             [3.0, 3.2, 3.1],
        'func_quality':          [0.90, 0.85, 0.88],
        'func_gsr':              [0.01, 0.02, 0.015],
    })
    path = tmp_path / "phenotypic.csv"
    df.to_csv(path, index=False)
    return path


# Used in: test_save_load_dataset.py
@pytest.fixture()
def dataset_components():
    """
    Small arrays and DataFrames for save_dataset / load_dataset tests.

    Returns a 6-tuple: (X_harmonised, X_raw, y, metadata, feature_labels, coords).
    """
    rng = np.random.default_rng(0)
    n_samples, n_features, n_rois = 10, 20, 10

    X_harmonised = rng.standard_normal((n_samples, n_features))
    X_raw = rng.standard_normal((n_samples, n_features))
    y = rng.integers(0, 2, n_samples)

    metadata = pd.DataFrame({
        'ID':          list(range(n_samples)),
        'SITE_ID':     ['NYU'] * n_samples,
        'DX_GROUP':    y.tolist(),
        'AGE_AT_SCAN': rng.uniform(10, 30, n_samples).round(2),
        'SEX':         rng.integers(1, 3, n_samples).tolist(),
    })

    feature_labels = pd.DataFrame({
        'feature_idx':  list(range(n_features)),
        'roi_i':        list(range(n_features)),
        'roi_j':        list(range(1, n_features + 1)),
        'label_i':      [f'Reg {i}' for i in range(n_features)],
        'label_j':      [f'Reg {i + 1}' for i in range(n_features)],
        'feature_name': [f'Reg {i} <-> Reg {i + 1}' for i in range(n_features)],
    })

    coords = pd.DataFrame({
        'x':     rng.uniform(-100, 100, n_rois).round(2),
        'y':     rng.uniform(-100, 100, n_rois).round(2),
        'z':     rng.uniform(-60,   80, n_rois).round(2),
        'label': [f'ROI_{i}' for i in range(n_rois)],
    })

    return X_harmonised, X_raw, y, metadata, feature_labels, coords


# Used in: test_centralised_logger.py
@pytest.fixture()
def sample_meta():
    """Minimal metadata dict for log_run."""
    return {
        'site':            'NYU',
        'sex':             'male',
        'n_participants':  100,
        'n_asd':           50,
        'n_td':            50,
        'eval_strategy':   'StratifiedKFold',
        'n_folds':         5,
    }


# Used in: test_centralised_logger.py
@pytest.fixture()
def sample_params():
    """Minimal hyperparameter dict for log_run."""
    return {
        'lr':      0.001,
        'epochs':  50,
        'dropout': 0.3,
    }


# Used in: test_centralised_logger.py
@pytest.fixture()
def sample_metrics():
    """Scalar metrics dict for log_run (no tensors or lists)."""
    return {
        'acc':      0.75,
        'auc':      0.80,
        'f1':       0.74,
        'sens':     0.72,
        'spec':     0.78,
        'std_acc':  0.03,
        'std_auc':  0.04,
        'std_f1':   0.05,
        'std_sens': 0.06,
        'std_spec': 0.02,
    }


# Used in: test_centralised_logger.py
@pytest.fixture()
def sample_artefacts():
    """
    Minimal artefacts dict for save_artefacts.

    'roc' contains numpy arrays (fpr, tpr, thresholds).
    'cm' is a 2x2 confusion matrix as a numpy array.
    Both are accepted by save_artefacts without requiring torch tensors.
    """
    fpr = np.array([0.0, 0.2, 0.5, 1.0])
    tpr = np.array([0.0, 0.6, 0.8, 1.0])
    thresholds = np.array([1.0, 0.8, 0.5, 0.0])
    cm = np.array([[40, 10], [12, 38]])
    return {'roc': (fpr, tpr, thresholds), 'cm': cm}
