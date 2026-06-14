"""
Fixtures for scripts tests.

Tier-2 fixtures: scoped to the scripts test directory.
"""

import importlib.util
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


def get_project_root() -> Path:
    """Resolve project root from this file's location (product/tests/scripts/)."""
    return Path(__file__).resolve().parents[3]


def _load_script(name: str, src: Path):
    """Load a script as a module, patching utils.paths.get_project_root so that
    module-level calls (e.g. ROOT = get_project_root() / 'product') resolve
    to the real repo root rather than any Google Drive mirror."""
    spec = importlib.util.spec_from_file_location(name, src)
    mod = importlib.util.module_from_spec(spec)
    with patch("utils.paths.get_project_root", return_value=get_project_root()):
        spec.loader.exec_module(mod)
    return mod


def _make_pheno_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal phenotypic DataFrame from a list of row dicts."""
    return pd.DataFrame(rows, columns=["SUB_ID", "SITE_ID", "AGE_AT_SCAN", "SEX"])


def _make_s3_key(site: str, sub_id: str) -> str:
    """Return a well-formed S3 key for the given site and subject ID."""
    return (
        f"data/Projects/ABIDE_Initiative/Outputs/cpac/"
        f"filt_noglobal/rois_cc200/{site}_{sub_id}_rois_cc200.1D"
    )


@pytest.fixture()
def make_pheno_df():
    """Return the _make_pheno_df helper so tests can build DataFrames without imports."""
    return _make_pheno_df


# Used in: test_download_participants.py (TestMainPhenoMapEdgeCases)
@pytest.fixture()
def make_s3_key():
    """Return the _make_s3_key helper so tests can build S3 keys without imports."""
    return _make_s3_key


# Used in: conftest.py (pheno_csv), test_download_participants.py
@pytest.fixture()
def module(tmp_path):
    """
    Import 02_download_participants with ROOT and EXT patched to tmp_path
    so that file-system side-effects are isolated.
    """
    root = get_project_root()
    mod = _load_script("dl_participants", root / "product/scripts/02_download_participants.py")
    mod.ROOT = tmp_path
    mod.EXT = tmp_path / "data" / "external"
    return mod


# Used in: test_download_participants.py
@pytest.fixture()
def pheno_csv(module, tmp_path):
    """Write a minimal phenotypic CSV to the patched EXT directory."""
    ext_dir = tmp_path / "data" / "external"
    ext_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ext_dir / module.PHENO_CSV

    df = _make_pheno_df([
        {"SUB_ID": "51001", "SITE_ID": "NYU",     "AGE_AT_SCAN": 12.0, "SEX": 1},
        {"SUB_ID": "51002", "SITE_ID": "NYU",     "AGE_AT_SCAN": 17.0, "SEX": 2},
        {"SUB_ID": "51003", "SITE_ID": "NYU",     "AGE_AT_SCAN": 25.0, "SEX": 1},
        {"SUB_ID": "51004", "SITE_ID": "YALE",    "AGE_AT_SCAN": 14.0, "SEX": 2},
        {"SUB_ID": "51005", "SITE_ID": "YALE",    "AGE_AT_SCAN": 30.0, "SEX": 1},
        {"SUB_ID": "51006", "SITE_ID": "TRINITY", "AGE_AT_SCAN": 19.0, "SEX": 2},
    ])
    df.to_csv(csv_path, index=False)
    return csv_path


# Used in: test_download_participants.py
@pytest.fixture()
def s3_keys_all():
    """S3 keys matching all six participants in pheno_csv."""
    return [
        _make_s3_key("NYU",     "0051001"),
        _make_s3_key("NYU",     "0051002"),
        _make_s3_key("NYU",     "0051003"),
        _make_s3_key("YALE",    "0051004"),
        _make_s3_key("YALE",    "0051005"),
        _make_s3_key("TRINITY", "0051006"),
    ]


# Used in: test_build_connectivity_dataset.py
@pytest.fixture()
def module_03():
    """Import 03_build_connectivity_dataset as a module for testing."""
    root = get_project_root()
    return _load_script("build_connectivity_dataset",
                        root / "product/scripts/03_build_connectivity_dataset.py")


# Used in: test_04_build_feature_labels.py
@pytest.fixture()
def module_04():
    """Import 04_build_feature_labels as a module for testing."""
    root = get_project_root()
    return _load_script("build_feature_labels",
                        root / "product/scripts/04_build_feature_labels.py")
