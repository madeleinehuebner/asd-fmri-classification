"""Tests for scripts/03_build_connectivity_dataset.py."""

import contextlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _run(module_03, tmp_path, **overrides):
    defaults = dict(
        raw_dir=str(tmp_path / "raw"),
        pheno_path=str(tmp_path / "pheno.csv"),
        processed_dir=str(tmp_path / "out"),
        prefix="test",
    )
    defaults.update(overrides)
    module_03.main(**defaults)


@pytest.fixture()
def pipeline_data():
    np.random.seed(7)
    pheno = pd.DataFrame({
        'SUB_ID':       [51001, 51002],
        'SITE_ID':      ['NYU', 'YALE'],
        'DX_GROUP':     [1, 2],
        'AGE_AT_SCAN':  [12.0, 18.0],
        'SEX':          [1, 2],
        'func_mean_fd': [0.10, 0.15],
        'FIQ':          [100.0, 110.0],
    })
    file_paths = ["/data/NYU_0051001_rois_cc200.1D", "/data/YALE_0051002_rois_cc200.1D"]
    ts_list    = [np.random.randn(100, 20), np.random.randn(100, 20)]
    local_ids  = ["NYU_0051001", "YALE_0051002"]
    X_conn     = np.random.randn(2, 10)
    X_harm     = X_conn * 1.1
    return dict(
        file_paths=file_paths, pheno=pheno,
        valid_paths=file_paths, ts_list=ts_list,
        local_ids=local_ids, X_conn=X_conn,
        X_matched=X_conn.copy(), metadata=pheno.copy(),
        X_harm=X_harm,
    )


@pytest.fixture()
def mocks(module_03, pipeline_data):
    d = pipeline_data
    m = {
        'load_timeseries':               MagicMock(return_value=d['file_paths']),
        'load_phenotypic_data':          MagicMock(return_value=d['pheno']),
        'save_dataset':                  MagicMock(),
        'preprocess_all':                MagicMock(return_value=(d['valid_paths'], d['ts_list'])),
        'extract_local_ids':             MagicMock(return_value=d['local_ids']),
        'match_participants':            MagicMock(return_value=(d['X_matched'], d['metadata'], None)),
        'filter_phenotypic_data':        MagicMock(return_value=d['pheno']),
        'match_males_to_females':        MagicMock(return_value=(d['pheno'], None)),
        'compute_connectivity_matrices': MagicMock(return_value=d['X_conn']),
        'apply_harmonisation':           MagicMock(return_value=d['X_harm']),
        'print_data_summary':            MagicMock(),
    }
    with contextlib.ExitStack() as stack:
        for name, mock_fn in m.items():
            stack.enter_context(patch.object(module_03, name, mock_fn))
        yield m


@pytest.fixture()
def female_meta_csv(tmp_path):
    df = pd.DataFrame({"ID": ["NYU_0051001", "YALE_0051002"]})
    path = tmp_path / "female_metadata.csv"
    df.to_csv(path, index=False)
    return path


class TestMainCallOrder:

    def test_all_pipeline_steps_called(self, module_03, mocks, tmp_path):
        """Every pipeline function is called exactly once in a standard run."""
        _run(module_03, tmp_path)
        for name in ('load_timeseries', 'load_phenotypic_data', 'preprocess_all',
                     'extract_local_ids', 'compute_connectivity_matrices',
                     'match_participants', 'apply_harmonisation',
                     'save_dataset', 'print_data_summary'):
            mocks[name].assert_called_once()

    def test_data_forwarded_correctly(self, module_03, mocks, pipeline_data, tmp_path):
        """ts_list reaches compute_connectivity_matrices; X_harm reaches print_data_summary."""
        _run(module_03, tmp_path)
        assert mocks['compute_connectivity_matrices'].call_args.args[0] is pipeline_data['ts_list']
        np.testing.assert_array_equal(
            mocks['print_data_summary'].call_args.args[0], pipeline_data['X_harm']
        )

    def test_load_functions_receive_correct_paths(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path)
        mocks['load_timeseries'].assert_called_once_with(str(tmp_path / "raw"))
        mocks['load_phenotypic_data'].assert_called_once_with(str(tmp_path / "pheno.csv"))


class TestMainSaveArguments:

    def _kwargs(self, mocks):
        return mocks['save_dataset'].call_args.kwargs

    def test_save_receives_arrays_and_metadata(self, module_03, mocks, pipeline_data, tmp_path):
        _run(module_03, tmp_path)
        kw = self._kwargs(mocks)
        np.testing.assert_array_equal(kw['X_harmonised'], pipeline_data['X_harm'])
        np.testing.assert_array_equal(kw['X_raw'], pipeline_data['X_matched'])
        assert not np.allclose(kw['X_harmonised'], kw['X_raw'])
        np.testing.assert_array_equal(kw['y'], pipeline_data['metadata']['DX_GROUP'].values)
        pd.testing.assert_frame_equal(kw['metadata'], pipeline_data['metadata'])

    def test_save_receives_correct_paths_and_prefix(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path)
        kw = self._kwargs(mocks)
        assert kw['processed_dir'] == Path(str(tmp_path / "out"))
        assert kw['prefix'] == "test"

    def test_custom_prefix_forwarded_and_no_feature_labels(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path, prefix="female")
        kw = self._kwargs(mocks)
        assert kw['prefix'] == "female"
        assert 'feature_labels' not in kw


class TestMainFilterBranch:

    def test_no_filters_skips_filter_phenotypic_data(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path)
        mocks['filter_phenotypic_data'].assert_not_called()
        mocks['match_males_to_females'].assert_not_called()

    def test_any_filter_calls_filter_phenotypic_data_once(self, module_03, mocks, tmp_path):
        """sex, age_min, age_max individually or together all trigger exactly one call."""
        _run(module_03, tmp_path, sex=1)
        mocks['filter_phenotypic_data'].assert_called_once()
        assert mocks['filter_phenotypic_data'].call_args[1].get('sex') == 1

    def test_age_filters_forwarded(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path, age_min=8.0, age_max=25.0)
        kw = mocks['filter_phenotypic_data'].call_args[1]
        assert kw.get('age_min') == 8.0
        assert kw.get('age_max') == 25.0

    def test_all_three_filters_call_filter_once(self, module_03, mocks, tmp_path):
        _run(module_03, tmp_path, sex=2, age_min=8.0, age_max=25.0)
        mocks['filter_phenotypic_data'].assert_called_once()
        mocks['match_males_to_females'].assert_not_called()


class TestMainMaleMatchingBranch:

    def test_match_males_called_with_correct_kwargs(
            self, module_03, mocks, pipeline_data, female_meta_csv, tmp_path):
        """match_males_to_females is called with female_ids, available_ids, and defaults."""
        _run(module_03, tmp_path, female_metadata_path=female_meta_csv)
        mocks['match_males_to_females'].assert_called_once()
        _, kwargs = mocks['match_males_to_females'].call_args
        assert kwargs['female_ids'] == {51001, 51002}
        assert kwargs['age_tolerance'] == 2.0
        assert kwargs['seed'] == 42
        expected_available = {int(id_.split("_")[-1]) for id_ in pipeline_data['local_ids']}
        assert kwargs['available_ids'] == expected_available

    def test_custom_age_tolerance_and_seed_forwarded(
            self, module_03, mocks, female_meta_csv, tmp_path):
        _run(module_03, tmp_path, female_metadata_path=female_meta_csv, age_tolerance=3.0, seed=99)
        _, kwargs = mocks['match_males_to_females'].call_args
        assert kwargs['age_tolerance'] == 3.0
        assert kwargs['seed'] == 99

    def test_filter_phenotypic_data_not_called_in_male_matching_branch(
            self, module_03, mocks, female_meta_csv, tmp_path):
        _run(module_03, tmp_path, female_metadata_path=female_meta_csv)
        mocks['filter_phenotypic_data'].assert_not_called()

    def test_preprocess_and_save_still_called(
            self, module_03, mocks, female_meta_csv, tmp_path):
        _run(module_03, tmp_path, female_metadata_path=female_meta_csv)
        mocks['preprocess_all'].assert_called_once()
        mocks['save_dataset'].assert_called_once()
