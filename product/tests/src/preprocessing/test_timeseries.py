"""Tests for preprocessing/timeseries.py."""

import numpy as np
import pytest

from preprocessing.timeseries import preprocess_all, preprocess_timeseries


class TestPreprocessTimeseries:

    def test_clean_input_returns_valid_2d_ndarray(self, clean_ts):
        """Clean input returns a 2-D ndarray with original ROI count, no NaN/inf."""
        result = preprocess_timeseries(clean_ts)
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape[1] == clean_ts.shape[1]
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_nan_rows_dropped(self, ts_with_nan_rows):
        """Rows 5, 20, 50 contain NaN → 97 valid rows remain."""
        result = preprocess_timeseries(ts_with_nan_rows)
        assert result is not None
        assert result.shape[0] == 97
        assert not np.any(np.isnan(result))

    def test_inf_rows_dropped(self, ts_with_inf_rows):
        """Rows 10 and 30 contain inf → 98 valid rows remain."""
        result = preprocess_timeseries(ts_with_inf_rows)
        assert result is not None
        assert result.shape[0] == 98
        assert not np.any(np.isinf(result))

    @pytest.mark.parametrize("n_rows,expected", [(30, None), (49, None), (50, "not_none")])
    def test_min_timepoints_threshold(self, n_rows, expected):
        """Below 50 rows → None; exactly 50 → accepted."""
        np.random.seed(3)
        ts = np.random.randn(n_rows, 20)
        result = preprocess_timeseries(ts)
        if expected is None:
            assert result is None
        else:
            assert result is not None

    def test_all_nan_returns_none(self):
        assert preprocess_timeseries(np.full((100, 20), np.nan)) is None

    def test_custom_min_timepoints(self):
        """min_timepoints parameter overrides the default of 50."""
        np.random.seed(3)
        ts = np.random.randn(40, 20)
        assert preprocess_timeseries(ts, min_timepoints=50) is None
        assert preprocess_timeseries(ts, min_timepoints=30) is not None

    def test_too_many_bad_rois_returns_none(self, ts_too_many_bad_rois):
        """8/20 = 40% bad ROIs exceeds max_bad_roi_ratio of 0.3."""
        assert preprocess_timeseries(ts_too_many_bad_rois) is None

    def test_bad_rois_zeroed_good_rois_preserved(self, ts_with_bad_rois):
        """Constant ROIs 0 and 10 are zeroed; remaining ROIs keep their values."""
        result = preprocess_timeseries(ts_with_bad_rois)
        assert result is not None
        assert np.all(result[:, 0] == 0.0)
        assert np.all(result[:, 10] == 0.0)
        assert not np.all(result[:, 1] == 0.0)

    def test_custom_max_bad_roi_ratio(self, ts_with_bad_rois, ts_too_many_bad_rois):
        """max_bad_roi_ratio parameter controls the rejection threshold."""
        assert preprocess_timeseries(ts_with_bad_rois, max_bad_roi_ratio=0.05) is None
        assert preprocess_timeseries(ts_too_many_bad_rois, max_bad_roi_ratio=0.5) is not None


class TestPreprocessAll:

    def test_returns_matched_lists_of_arrays(self, ts_1d_files):
        all_paths, _ = ts_1d_files
        valid_paths, cleaned_ts = preprocess_all(all_paths)
        assert isinstance(valid_paths, list)
        assert isinstance(cleaned_ts, list)
        assert len(valid_paths) == len(cleaned_ts)
        assert all(isinstance(ts, np.ndarray) for ts in cleaned_ts)

    def test_invalid_files_excluded_valid_order_preserved(self, ts_1d_files):
        """Too-short file is excluded; valid files keep input order."""
        all_paths, n_valid = ts_1d_files
        valid_paths, cleaned_ts = preprocess_all(all_paths)
        assert len(cleaned_ts) == n_valid
        assert all(p in all_paths for p in valid_paths)
        expected_order = [p for p in all_paths if p in set(valid_paths)]
        assert valid_paths == expected_order

    def test_output_arrays_meet_quality_requirements(self, ts_1d_files):
        """All returned arrays have correct shape and no NaN/inf."""
        all_paths, _ = ts_1d_files
        _, cleaned_ts = preprocess_all(all_paths)
        assert all(ts.shape[1] == 20 for ts in cleaned_ts)
        assert all(ts.shape[0] >= 50 for ts in cleaned_ts)
        assert all(not np.any(np.isnan(ts)) for ts in cleaned_ts)
        assert all(not np.any(np.isinf(ts)) for ts in cleaned_ts)

    def test_empty_input_returns_empty_lists(self):
        valid_paths, cleaned_ts = preprocess_all([])
        assert valid_paths == []
        assert cleaned_ts == []
