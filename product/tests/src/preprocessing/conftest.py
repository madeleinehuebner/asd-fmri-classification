"""
Fixtures for src/preprocessing tests.

Tier-2 fixtures: scoped to the preprocessing test directory.
"""

import pytest
import numpy as np
import pandas as pd


# Used in: test_timeseries.py
@pytest.fixture()
def clean_ts():
    """Well-formed timeseries: 100 timepoints * 20 ROIs, no NaN/inf, good variance in all ROIs."""
    np.random.seed(1)
    return np.random.randn(100, 20)


# Used in: test_timeseries.py
@pytest.fixture()
def ts_with_nan_rows():
    """Timeseries where rows 5, 20, and 50 contain NaN values; 97 valid rows remain."""
    np.random.seed(1)
    ts = np.random.randn(100, 20)
    ts[5, :]  = np.nan   # entire row
    ts[20, 3] = np.nan   # single element
    ts[50, :] = np.nan   # entire row
    return ts


# Used in: test_timeseries.py
@pytest.fixture()
def ts_with_inf_rows():
    """Timeseries where rows 10 and 30 contain infinite values; 98 valid rows remain."""
    np.random.seed(1)
    ts = np.random.randn(100, 20)
    ts[10, 0] = np.inf
    ts[30, :]  = -np.inf
    return ts


# Used in: test_timeseries.py
@pytest.fixture()
def ts_too_short():
    """Timeseries with only 30 valid timepoints below the default min_timepoints of 50."""
    np.random.seed(1)
    return np.random.randn(30, 20)


# Used in: test_timeseries.py
@pytest.fixture()
def ts_with_bad_rois():
    """Timeseries where ROIs 0 and 10 are constant (zero variance): 2/20 = 10% bad, below the 0.3 
    threshold."""
    np.random.seed(1)
    ts = np.random.randn(100, 20)
    ts[:, 0]  = 5.0    # constant → variance = 0
    ts[:, 10] = -3.0   # constant → variance = 0
    return ts


# Used in: test_timeseries.py
@pytest.fixture()
def ts_too_many_bad_rois():
    """Timeseries where ROIs 0-7 are constant (zero variance): 8/20 = 40% bad, above the 0.3 
    threshold."""
    np.random.seed(1)
    ts = np.random.randn(100, 20)
    for i in range(8):
        ts[:, i] = float(i)   # constant → variance = 0
    return ts


# Used in: test_timeseries.py
@pytest.fixture()
def ts_1d_files(tmp_path):
    """
    Temp directory containing three .1D files:
        - valid_00.1D, valid_01.1D  (100 * 20 random pass preprocessing)
        - invalid_00.1D             (30  * 20 random fails: too few timepoints)
    Returns (all_paths, n_valid) where all_paths is sorted alphabetically.
    """
    np.random.seed(2)
    for i in range(2):
        p = tmp_path / f"valid_{i:02d}.1D"
        np.savetxt(str(p), np.random.randn(100, 20))
    bad = tmp_path / "invalid_00.1D"
    np.savetxt(str(bad), np.random.randn(30, 20))
    all_paths = sorted([str(p) for p in tmp_path.glob("*.1D")])
    return all_paths, 2


# Used in: test_participants.py, test_harmonisation.py
@pytest.fixture()
def participants_pheno_df():
    """
    Minimal phenotypic DataFrame with 6 participants across 3 sites.

    Subjects and properties:
      50952  NYU      ASD(1)     age 12  Male(1)   fd 0.10
      51234  NYU      Control(2) age 13  Female(2) fd 0.15
      51100  NYU      ASD(1)     age 20  Male(1)   fd 0.20
      51300  YALE     Control(2) age 16  Male(1)   fd 0.12
      51400  YALE     ASD(1)     age 17  Female(2) fd 0.18
      51500  CALTECH  Control(2) age 22  Female(2) fd NaN
    """
    return pd.DataFrame({
        'subject':      [50952, 51234, 51100, 51300, 51400, 51500],
        'participant':  [50952, 51234, 51100, 51300, 51400, 51500],
        'SITE_ID':      ['NYU', 'NYU', 'NYU', 'YALE', 'YALE', 'CALTECH'],
        'DX_GROUP':     [1, 2, 1, 2, 1, 2],
        'AGE_AT_SCAN':  [12.0, 13.0, 20.0, 16.0, 17.0, 22.0],
        'SEX':          [1, 2, 1, 1, 2, 2],
        'func_mean_fd': [0.10, 0.15, 0.20, 0.12, 0.18, np.nan],
    })


# Used in: test_participants.py
@pytest.fixture()
def participants_file_paths():
    """
    Six ABIDE-style file paths in scrambled subject order.
    Basenames follow the pattern <SITE>_<SUBJECT_ID>_rois_cc200.1D.
    Order (index in X): CALTECH_0051500=0, YALE_0051400=1, NYU_0050952=2,
                        YALE_0051300=3, NYU_0051234=4, NYU_0051100=5.
    """
    return [
        '/data/raw/CALTECH_0051500_rois_cc200.1D',
        '/data/raw/YALE_0051400_rois_cc200.1D',
        '/data/raw/NYU_0050952_rois_cc200.1D',
        '/data/raw/YALE_0051300_rois_cc200.1D',
        '/data/raw/NYU_0051234_rois_cc200.1D',
        '/data/raw/NYU_0051100_rois_cc200.1D',
    ]


# Used in: test_participants.py
@pytest.fixture()
def participants_local_ids():
    """
    Local IDs extracted from participants_file_paths in the same scrambled order.
    Maps to X rows: [CALTECH_0051500, YALE_0051400, NYU_0050952,
                     YALE_0051300, NYU_0051234, NYU_0051100].
    """
    return [
        'CALTECH_0051500',
        'YALE_0051400',
        'NYU_0050952',
        'YALE_0051300',
        'NYU_0051234',
        'NYU_0051100',
    ]


# Used in: test_participants.py
@pytest.fixture()
def participants_X():
    """Feature matrix (6 subjects * 10 features) aligned to participants_local_ids order."""
    np.random.seed(5)
    return np.random.randn(6, 10)


# Used in: test_connectivity.py
@pytest.fixture()
def cc200_csv(tmp_path):
    """
    Synthetic CC200_ROI_labels.csv with 20 ROIs in the expected AAL format.

    The 'AAL' column contains strings like '["Region_1": 0.90]' so that
    load_roi_labels() can extract region names via regex.  Row count matches
    clean_ts (100 × 20) so both fixtures can be used together in
    compute_connectivity_matrices tests without dimension mismatches.

    Returns (path_str, n_rois) so tests can derive expected feature counts
    without hardcoding the ROI number.
    """
    n_rois = 20
    df = pd.DataFrame({
        'AAL': [f'["Region_{i}": 0.90]' for i in range(1, n_rois + 1)],
    })
    path = tmp_path / "CC200_ROI_labels.csv"
    df.to_csv(path, index=False)
    return str(path), n_rois


# Used in: test_connectivity.py
@pytest.fixture()
def aal_map_csv(tmp_path):
    """
    Synthetic AAL_map.csv with 20 entries matching the regions in cc200_csv.

    LABEL values ('Region_1' … 'Region_20') align with the region names
    extracted from cc200_csv so the merge inside load_roi_labels() succeeds
    and every row is fully populated (no 'Unknown Region' fallbacks).

    Returns (path_str, n_rois).
    """
    n_rois = 20
    df = pd.DataFrame({
        'LABEL':                  [f'Region_{i}' for i in range(1, n_rois + 1)],
        'ANATOMICAL DESCRIPTION': [f'Region {i} Desc' for i in range(1, n_rois + 1)],
        'NOTATION':               [f'Reg {i}' for i in range(1, n_rois + 1)],
        'ABBREVIATION':           [f'R{i}' for i in range(1, n_rois + 1)],
        'NETWORK':                ['Default Mode'] * n_rois,
    })
    path = tmp_path / "AAL_map.csv"
    df.to_csv(path, index=False)
    return str(path), n_rois


# Used in: test_harmonisation.py
@pytest.fixture()
def harmonisation_data():
    """
    Synthetic (X, metadata) pair for apply_harmonisation / prepare_combat_covariates tests.

    Design: 3 sites * 8 subjects = 24 total, 50 features.
      - Each site has 4 control (DX_GROUP=0) and 4 ASD (DX_GROUP=1) subjects.
      - Sex alternates 1/2 within each site so ComBat's design matrix is full-rank.
      - Per-site, per-feature offsets simulate realistic batch effects for ComBat to correct.

    Uses isolated RandomState objects so the global numpy seed is not mutated.
    """
    n_per_site = 8
    site_names = ['NYU', 'YALE', 'CALTECH']
    n = n_per_site * len(site_names)
    n_features = 50

    site_list = [s for s in site_names for _ in range(n_per_site)]

    rng = np.random.default_rng(seed=42)
    X = rng.standard_normal((n, n_features))

    # Independent per-site, per-feature offsets (different seed per site)
    for k, site in enumerate(site_names):
        rng = np.random.default_rng(100 + k)
        offset = rng.standard_normal(n_features) * 0.5
        mask = [s == site for s in site_list]
        X[mask] += offset

    metadata = pd.DataFrame({
        'SITE_ID':      site_list,
        'DX_GROUP':     [0, 0, 0, 0, 1, 1, 1, 1] * len(site_names),
        'AGE_AT_SCAN':  [10.0 + i * 2 for i in range(n_per_site)] * len(site_names),
        'SEX':          [1, 2, 1, 2, 1, 2, 1, 2] * len(site_names),
        'func_mean_fd': [0.10 + 0.02 * i for i in range(n_per_site)] * len(site_names),
    })

    return X, metadata
