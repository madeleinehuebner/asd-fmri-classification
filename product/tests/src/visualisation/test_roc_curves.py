"""Tests for visualisation/roc_curves.py."""

import numpy as np
import pytest

from visualisation.roc_curves import interpolate_roc, site_label


class TestInterpolateRoc:

    def test_output_fpr_is_linspace(self):
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.7, 1.0])
        fpr_grid, _ = interpolate_roc(fpr, tpr)
        np.testing.assert_array_almost_equal(fpr_grid, np.linspace(0, 1, 200))

    def test_output_shapes_match(self):
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.7, 1.0])
        fpr_grid, tpr_interp = interpolate_roc(fpr, tpr)
        assert len(fpr_grid) == len(tpr_interp) == 200

    def test_custom_n(self):
        fpr = np.array([0.0, 0.5, 1.0])
        tpr = np.array([0.0, 0.7, 1.0])
        fpr_grid, tpr_interp = interpolate_roc(fpr, tpr, n=50)
        assert len(fpr_grid) == len(tpr_interp) == 50

    def test_perfect_curve_tpr_near_one(self):
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        _, tpr_interp = interpolate_roc(fpr, tpr)
        # All interpolated TPR values after fpr=0 should be 1.0
        assert np.all(tpr_interp[1:] == pytest.approx(1.0))


class TestSiteLabel:

    def test_plain_filename(self):
        assert site_label("NYU_fold_0.json") == "NYU"

    def test_site_prefix(self):
        assert site_label("site_NYU_fold.json") == "NYU"

    def test_single_token(self):
        assert site_label("YALE") == "YALE"

    def test_space_separated(self):
        assert site_label("NYU fold_0") == "NYU"
