"""Tests for scripts/04_build_feature_labels.py."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _run(module_04, tmp_path, **overrides):
    defaults = dict(
        cc200_map=str(tmp_path / "cc200.csv"),
        aal_map=str(tmp_path / "aal.csv"),
        atlas=str(tmp_path / "atlas.nii.gz"),
        processed_dir=str(tmp_path / "processed"),
    )
    defaults.update(overrides)
    module_04.main(**defaults)


@pytest.fixture()
def label_data():
    feature_df = pd.DataFrame({
        'feature_idx':  list(range(5)),
        'roi_i':        list(range(5)),
        'roi_j':        list(range(1, 6)),
        'feature_name': [f'ROI_{i} <-> ROI_{i+1}' for i in range(5)],
    })
    coords_df = pd.DataFrame({
        'x': [0.0, 1.0, 2.0], 'y': [0.0, 1.0, 2.0], 'z': [0.0, 1.0, 2.0],
        'label': ['ROI_0', 'ROI_1', 'ROI_2'],
    })
    return dict(feature_df=feature_df, coords_df=coords_df)


@pytest.fixture()
def mocks(module_04, label_data):
    d = label_data
    m = {
        'generate_feature_labels': MagicMock(return_value=d['feature_df']),
        'generate_coords_labels':  MagicMock(return_value=d['coords_df']),
        'save_dataset':            MagicMock(),
    }
    with contextlib.ExitStack() as stack:
        for name, mock_fn in m.items():
            stack.enter_context(patch.object(module_04, name, mock_fn))
        yield m


class TestMainCallOrder:

    def test_all_pipeline_steps_called_once(self, module_04, mocks, tmp_path):
        _run(module_04, tmp_path)
        for name in ('generate_feature_labels', 'generate_coords_labels', 'save_dataset'):
            mocks[name].assert_called_once()


class TestMainGenerateArguments:

    def test_generate_feature_labels_receives_correct_args(self, module_04, mocks, tmp_path):
        """Hardcoded ROI count is 200; cc200 and aal paths are forwarded."""
        _run(module_04, tmp_path)
        args = mocks['generate_feature_labels'].call_args.args
        assert args[0] == 200
        assert args[1] == str(tmp_path / "cc200.csv")
        assert args[2] == str(tmp_path / "aal.csv")

    def test_generate_coords_labels_receives_correct_args(self, module_04, mocks, tmp_path):
        _run(module_04, tmp_path)
        args = mocks['generate_coords_labels'].call_args.args
        assert args[0] == str(tmp_path / "cc200.csv")
        assert args[1] == str(tmp_path / "aal.csv")
        assert args[2] == str(tmp_path / "atlas.nii.gz")

    def test_custom_paths_forwarded(self, module_04, mocks, tmp_path):
        custom_cc200 = str(tmp_path / "custom_cc200.csv")
        custom_atlas = str(tmp_path / "my_atlas.nii.gz")
        _run(module_04, tmp_path, cc200_map=custom_cc200, atlas=custom_atlas)
        assert mocks['generate_feature_labels'].call_args.args[1] == custom_cc200
        assert mocks['generate_coords_labels'].call_args.args[2] == custom_atlas


class TestMainSaveArguments:

    def _kwargs(self, mocks):
        return mocks['save_dataset'].call_args.kwargs

    def test_save_receives_dataframes_from_generate(
            self, module_04, mocks, label_data, tmp_path):
        """save_dataset gets the DataFrames returned by the generate functions."""
        _run(module_04, tmp_path)
        kw = self._kwargs(mocks)
        pd.testing.assert_frame_equal(kw['feature_labels'], label_data['feature_df'])
        pd.testing.assert_frame_equal(kw['coords'], label_data['coords_df'])

    def test_save_receives_processed_dir(self, module_04, mocks, tmp_path):
        _run(module_04, tmp_path)
        assert self._kwargs(mocks)['processed_dir'] == Path(str(tmp_path / "processed"))

    def test_custom_processed_dir_forwarded(self, module_04, mocks, tmp_path):
        custom = str(tmp_path / "my_output")
        _run(module_04, tmp_path, processed_dir=custom)
        assert self._kwargs(mocks)['processed_dir'] == Path(custom)

    def test_array_keys_not_passed_to_save(self, module_04, mocks, tmp_path):
        """04 only saves labels; it must not pass X, y, or metadata."""
        _run(module_04, tmp_path)
        kw = self._kwargs(mocks)
        for key in ('X_harmonised', 'X_raw', 'y', 'metadata'):
            assert key not in kw
