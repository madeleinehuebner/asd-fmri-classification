"""Tests for preprocessing/harmonisation.py."""

import numpy as np
import pandas as pd
import pytest

from preprocessing.harmonisation import apply_harmonisation, prepare_combat_covariates


class TestPrepareCombatCovariates:

    def test_output_structure(self, participants_pheno_df):
        """Returns a DataFrame with the correct columns and row count."""
        covars = prepare_combat_covariates(participants_pheno_df)
        assert isinstance(covars, pd.DataFrame)
        assert list(covars.columns) == ['batch', 'diagnosis', 'age', 'sex', 'motion']
        assert len(covars) == len(participants_pheno_df)

    def test_column_mappings(self, participants_pheno_df):
        """batch=SITE_ID, diagnosis=DX_GROUP, age=AGE_AT_SCAN, sex=SEX."""
        covars = prepare_combat_covariates(participants_pheno_df)
        np.testing.assert_array_equal(covars['batch'].values,     participants_pheno_df['SITE_ID'].values)
        np.testing.assert_array_equal(covars['diagnosis'].values, participants_pheno_df['DX_GROUP'].values)
        np.testing.assert_array_equal(covars['age'].values,       participants_pheno_df['AGE_AT_SCAN'].values)
        np.testing.assert_array_equal(covars['sex'].values,       participants_pheno_df['SEX'].values)

    def test_nan_motion_filled_with_column_mean(self, participants_pheno_df):
        """The NaN in func_mean_fd is replaced with the column mean; no NaN remain."""
        expected_fill = participants_pheno_df['func_mean_fd'].mean()
        nan_idx = participants_pheno_df['func_mean_fd'].isna().to_numpy().nonzero()[0][0]
        covars = prepare_combat_covariates(participants_pheno_df)
        assert covars['motion'].iloc[nan_idx] == pytest.approx(expected_fill)
        assert not covars['motion'].isna().any()
        assert np.all(np.isfinite(covars['motion'].values))

    def test_non_nan_motion_preserved(self, participants_pheno_df):
        non_nan = ~participants_pheno_df['func_mean_fd'].isna()
        covars = prepare_combat_covariates(participants_pheno_df)
        np.testing.assert_array_almost_equal(
            covars['motion'][non_nan.values].values,
            participants_pheno_df['func_mean_fd'][non_nan].values,
        )

    def test_numeric_dtypes(self, participants_pheno_df):
        covars = prepare_combat_covariates(participants_pheno_df)
        assert covars['age'].dtype in [np.float32, np.float64]
        assert covars['motion'].dtype in [np.float32, np.float64]


class TestApplyHarmonisation:

    def test_output_shape_and_type(self, harmonisation_data):
        X, meta = harmonisation_data
        X_harm = apply_harmonisation(X, meta)
        assert isinstance(X_harm, np.ndarray)
        assert X_harm.shape == X.shape

    def test_output_is_finite_and_differs_from_input(self, harmonisation_data):
        """ComBat modifies the data; output must be finite and not identical to input."""
        X, meta = harmonisation_data
        X_harm = apply_harmonisation(X, meta)
        assert np.all(np.isfinite(X_harm))
        assert not np.allclose(X_harm, X)

    def test_biological_signal_preserved(self, harmonisation_data):
        """Per-subject correlation between raw and harmonised should be strong (> 0.5)."""
        X, meta = harmonisation_data
        X_harm = apply_harmonisation(X, meta)
        correlations = [np.corrcoef(X[i], X_harm[i])[0, 1] for i in range(len(X))]
        assert np.mean(correlations) > 0.5
