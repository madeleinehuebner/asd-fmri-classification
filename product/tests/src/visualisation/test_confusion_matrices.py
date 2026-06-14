"""Tests for visualisation/confusion_matrices.py."""

import pytest

from visualisation.confusion_matrices import default_filename, default_title


class TestDefaultTitle:

    def test_skf_cv_label(self):
        assert "Stratified 5-Fold CV" in default_title("SVM + PCA", "SKF", "Baseline")

    def test_loso_cv_label(self):
        assert "Leave-One-Site-Out CV" in default_title("SVM + PCA", "LOSO", "Baseline")

    def test_contains_label(self):
        assert "RF + SDAE" in default_title("RF + SDAE", "SKF", "Optimised")

    def test_contains_experiment(self):
        assert "Optimised" in default_title("RF + SDAE", "SKF", "Optimised")


class TestDefaultFilename:

    def test_spaces_become_underscores(self):
        result = default_filename("RF PCA", "SKF", "Baseline")
        assert " " not in result
        assert "rf_pca" in result

    def test_plus_removed(self):
        result = default_filename("RF + PCA", "SKF", "Baseline")
        assert "+" not in result

    def test_contains_cv_lowercased(self):
        assert "skf" in default_filename("RF + PCA", "SKF", "Baseline")
        assert "loso" in default_filename("RF + PCA", "LOSO", "Baseline")

    def test_contains_experiment_lowercased(self):
        assert "optimised" in default_filename("RF + PCA", "SKF", "Optimised")

    def test_pdf_extension(self):
        assert default_filename("RF + PCA", "SKF", "Baseline").endswith(".pdf")
