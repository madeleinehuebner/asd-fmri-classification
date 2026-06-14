"""Tests for visualisation/common.py."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from visualisation.common import auc_from_roc, save_figure


class TestAucFromRoc:

    def test_perfect_classifier(self):
        fpr = np.array([0.0, 0.0, 1.0])
        tpr = np.array([0.0, 1.0, 1.0])
        assert auc_from_roc(fpr, tpr) == pytest.approx(1.0)

    def test_random_classifier(self):
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)
        assert auc_from_roc(fpr, tpr) == pytest.approx(0.5, abs=1e-3)

    def test_unsorted_fpr_matches_sorted(self):
        fpr = np.array([1.0, 0.0, 0.5])
        tpr = np.array([1.0, 0.0, 0.8])
        fpr_sorted = np.array([0.0, 0.5, 1.0])
        tpr_sorted = np.array([0.0, 0.8, 1.0])
        assert auc_from_roc(fpr, tpr) == pytest.approx(auc_from_roc(fpr_sorted, tpr_sorted))

    def test_single_point_returns_zero(self):
        fpr = np.array([0.5])
        tpr = np.array([0.5])
        assert auc_from_roc(fpr, tpr) == pytest.approx(0.0)


class TestSaveFigure:

    def test_save_false_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("visualisation.common.figures_dir", lambda: tmp_path)
        fig, _ = plt.subplots()
        save_figure(fig, "test.pdf", save=False)
        assert not (tmp_path / "test.pdf").exists()
        plt.close(fig)

    def test_save_true_writes_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("visualisation.common.figures_dir", lambda: tmp_path)
        fig, _ = plt.subplots()
        save_figure(fig, "test.pdf", save=True)
        assert (tmp_path / "test.pdf").exists()

    def test_save_true_creates_directory(self, tmp_path, monkeypatch):
        subdir = tmp_path / "nested" / "figures"
        monkeypatch.setattr("visualisation.common.figures_dir", lambda: subdir)
        subdir.mkdir(parents=True, exist_ok=True)
        fig, _ = plt.subplots()
        save_figure(fig, "out.pdf", save=True)
        assert (subdir / "out.pdf").exists()
