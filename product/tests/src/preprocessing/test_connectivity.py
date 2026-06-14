"""Tests for preprocessing/connectivity.py."""

import numpy as np
import pandas as pd
import pytest

from preprocessing.connectivity import (
    compute_connectivity_matrices,
    generate_feature_labels,
    load_roi_labels,
)


class TestLoadRoiLabels:

    def test_returns_dataframe_with_required_columns(self, cc200_csv, aal_map_csv):
        cc200_path, n_rois = cc200_csv
        aal_path, _ = aal_map_csv
        df = load_roi_labels(cc200_path, aal_path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == n_rois
        for col in ('AAL', 'NOTATION', 'ABBREVIATION'):
            assert col in df.columns

    def test_no_nan_in_notation_or_abbreviation(self, cc200_csv, aal_map_csv):
        cc200_path, _ = cc200_csv
        aal_path, _ = aal_map_csv
        df = load_roi_labels(cc200_path, aal_path)
        assert df['NOTATION'].notna().all()
        assert df['ABBREVIATION'].notna().all()

    def test_unknown_region_fallback(self, tmp_path, aal_map_csv):
        """An ROI whose AAL label has no match in aal_map falls back to 'Unknown Region'."""
        aal_path, _ = aal_map_csv
        df = pd.DataFrame({'AAL': ['["Region_1": 0.90]', '["None": 0.50]']})
        cc200_path = str(tmp_path / "cc200_partial.csv")
        df.to_csv(cc200_path, index=False)
        result = load_roi_labels(cc200_path, aal_path)
        assert (result['NOTATION'] == 'Unknown Region').any()


class TestGenerateFeatureLabels:

    def test_returns_dataframe_with_required_columns(self, cc200_csv, aal_map_csv):
        cc200_path, n_rois = cc200_csv
        aal_path, _ = aal_map_csv
        df = generate_feature_labels(n_rois, cc200_path, aal_path)
        assert isinstance(df, pd.DataFrame)
        for col in ('feature_idx', 'roi_i', 'roi_j', 'label_i', 'label_j', 'feature_name'):
            assert col in df.columns

    def test_correct_feature_count_and_upper_triangle(self, cc200_csv, aal_map_csv):
        """Feature count equals n_rois*(n_rois-1)//2; all pairs satisfy roi_i < roi_j."""
        cc200_path, n_rois = cc200_csv
        aal_path, _ = aal_map_csv
        df = generate_feature_labels(n_rois, cc200_path, aal_path)
        assert len(df) == n_rois * (n_rois - 1) // 2
        assert (df['roi_i'] < df['roi_j']).all()
        assert df['feature_idx'].tolist() == list(range(len(df)))

    def test_label_columns_and_feature_names(self, cc200_csv, aal_map_csv):
        """label_i/j come from NOTATION; feature_name uses AAL strings with ' <-> ' separator."""
        cc200_path, n_rois = cc200_csv
        aal_path, _ = aal_map_csv
        df = generate_feature_labels(n_rois, cc200_path, aal_path)
        labels_df = load_roi_labels(cc200_path, aal_path)
        first = df.iloc[0]
        assert first['label_i'] == labels_df.iloc[int(first['roi_i'])]['NOTATION']
        assert first['label_j'] == labels_df.iloc[int(first['roi_j'])]['NOTATION']
        assert ' <-> ' in first['feature_name']
        assert df['feature_name'].str.contains(' <-> ').all()


class TestComputeConnectivityMatrices:

    def test_output_shape_and_type(self, clean_ts, cc200_csv):
        _, n_rois = cc200_csv
        np.random.seed(7)
        ts2 = np.random.randn(100, n_rois)
        X = compute_connectivity_matrices([clean_ts, ts2])
        assert isinstance(X, np.ndarray)
        assert X.shape == (2, n_rois * (n_rois - 1) // 2)
        assert np.all(np.isfinite(X))

    def test_deterministic_and_batch_equals_individual(self, clean_ts, cc200_csv):
        """Same input → same output; batch and individual produce identical results."""
        _, n_rois = cc200_csv
        np.random.seed(7)
        ts2 = np.random.randn(100, n_rois)

        X_batch = compute_connectivity_matrices([clean_ts, ts2])
        np.testing.assert_array_equal(
            X_batch, compute_connectivity_matrices([clean_ts, ts2])
        )
        np.testing.assert_array_almost_equal(
            X_batch[0], compute_connectivity_matrices([clean_ts])[0]
        )
        np.testing.assert_array_almost_equal(
            X_batch[1], compute_connectivity_matrices([ts2])[0]
        )
