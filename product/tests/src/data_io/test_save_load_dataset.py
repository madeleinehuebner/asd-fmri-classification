"""Tests for data_io/save_load_dataset.py."""

import numpy as np
import pandas as pd
import pytest

from data_io.save_load_dataset import (
    load_dataset,
    load_phenotypic_data,
    load_timeseries,
    save_dataset,
)


class TestLoadTimeseries:

    def test_finds_1d_files_sorted(self, sample_1d_dir):
        """Returns exactly the .1D files, sorted, ignoring other extensions."""
        paths = load_timeseries(str(sample_1d_dir))
        assert len(paths) == 3
        assert all(p.endswith(".1D") for p in paths)
        assert paths == sorted(paths)
        assert all(isinstance(p, str) for p in paths)

    def test_empty_dir_returns_empty_list(self, tmp_path):
        assert load_timeseries(str(tmp_path)) == []


class TestLoadPhenotypicData:

    def test_returns_dataframe_with_expected_columns_and_rows(self, sample_pheno_csv_path):
        df = load_phenotypic_data(str(sample_pheno_csv_path))
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        expected = [
            'subject', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX',
            'FIQ', 'VIQ', 'PIQ', 'HANDEDNESS_CATEGORY', 'EYE_STATUS_AT_SCAN',
            'func_mean_fd', 'func_num_fd', 'func_perc_fd',
            'func_dvars', 'func_outlier', 'func_efc', 'func_fber',
            'func_fwhm', 'func_quality', 'func_gsr',
        ]
        assert list(df.columns) == expected

    def test_custom_columns(self, sample_pheno_csv_path):
        cols = ['subject', 'SITE_ID', 'DX_GROUP']
        df = load_phenotypic_data(str(sample_pheno_csv_path), columns=cols)
        assert list(df.columns) == cols

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(Exception):
            load_phenotypic_data(str(tmp_path / "nonexistent.csv"))


class TestSaveDataset:

    def test_npz_and_metadata_created_with_correct_content(self, tmp_path, dataset_components):
        X_harm, X_raw, y, meta, feat, coords = dataset_components
        save_dataset(tmp_path, X_harm, X_raw, y, meta)
        npz_path = tmp_path / "abide" / "abide_harmonised.npz"
        assert npz_path.exists()
        assert (tmp_path / "abide" / "abide_metadata.csv").exists()
        data = np.load(npz_path)
        assert set(data.files) == {"X", "X_raw", "y"}

    def test_feature_labels_and_coords_created(self, tmp_path, dataset_components):
        _, _, _, _, feat, coords = dataset_components
        save_dataset(tmp_path, feature_labels=feat, coords=coords)
        assert (tmp_path / "feature_labels.csv").exists()
        assert (tmp_path / "coords.csv").exists()

    def test_custom_prefix(self, tmp_path, dataset_components):
        X_harm, X_raw, y, meta, *_ = dataset_components
        save_dataset(tmp_path, X_harm, X_raw, y, meta, prefix="custom")
        assert (tmp_path / "custom" / "custom_harmonised.npz").exists()
        assert (tmp_path / "custom" / "custom_metadata.csv").exists()

    def test_missing_required_args_prevents_write(self, tmp_path, dataset_components):
        """npz is not written if metadata is None; feature_labels.csv not written if coords is None."""
        X_harm, X_raw, y, meta, feat, coords = dataset_components
        save_dataset(tmp_path, X_harm, X_raw, y, metadata=None)
        assert not (tmp_path / "abide" / "abide_harmonised.npz").exists()
        save_dataset(tmp_path, feature_labels=feat, coords=None)
        assert not (tmp_path / "feature_labels.csv").exists()

    def test_creates_missing_directory(self, tmp_path, dataset_components):
        X_harm, X_raw, y, meta, *_ = dataset_components
        new_dir = tmp_path / "nested" / "dir"
        save_dataset(new_dir, X_harm, X_raw, y, meta)
        assert new_dir.is_dir()


class TestLoadDataset:

    @pytest.fixture()
    def saved(self, tmp_path, dataset_components, monkeypatch):
        X_harm, X_raw, y, meta, feat, coords = dataset_components
        processed_dir = tmp_path / "product" / "data" / "processed"
        save_dataset(processed_dir, X_harm, X_raw, y, meta,
                     feature_labels=feat, coords=coords, prefix="test")
        monkeypatch.setattr("data_io.save_load_dataset.get_project_root", lambda: tmp_path)
        return dataset_components

    def test_roundtrip_arrays_and_metadata(self, saved):
        X_harm, X_raw, y, meta, feat, coords = saved
        X_h, X_r, y_l, meta_l, feat_l = load_dataset(prefix="test")
        np.testing.assert_array_equal(X_h, X_harm)
        np.testing.assert_array_equal(X_r, X_raw)
        np.testing.assert_array_equal(y_l, y)
        pd.testing.assert_frame_equal(meta_l, meta, check_dtype=False)
        pd.testing.assert_frame_equal(feat_l, feat, check_dtype=False)

    def test_missing_files_raise(self, saved, tmp_path):
        base = tmp_path / "product" / "data" / "processed"
        (base / "test" / "test_harmonised.npz").unlink()
        with pytest.raises(FileNotFoundError):
            load_dataset(prefix="test")
