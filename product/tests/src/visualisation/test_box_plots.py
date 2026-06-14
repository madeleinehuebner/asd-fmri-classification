"""Tests for visualisation/box_plots.py."""

import numpy as np
import pytest

from visualisation.box_plots import extract_metrics, metrics_from_cm


class TestMetricsFromCm:

    def test_perfect_cm(self):
        cm = np.array([[50, 0], [0, 50]])
        m = metrics_from_cm(cm)
        assert m["acc"]  == pytest.approx(1.0)
        assert m["sens"] == pytest.approx(1.0)
        assert m["spec"] == pytest.approx(1.0)
        assert m["f1"]   == pytest.approx(1.0)

    def test_all_wrong_cm(self):
        cm = np.array([[0, 50], [50, 0]])
        m = metrics_from_cm(cm)
        assert m["acc"]  == pytest.approx(0.0)
        assert m["sens"] == pytest.approx(0.0)
        assert m["spec"] == pytest.approx(0.0)

    def test_zero_tp_fp_gives_nan_f1(self):
        cm = np.array([[50, 0], [50, 0]])
        m = metrics_from_cm(cm)
        assert np.isnan(m["f1"])

    def test_zero_tn_fp_gives_nan_spec(self):
        cm = np.array([[0, 0], [10, 40]])
        m = metrics_from_cm(cm)
        assert np.isnan(m["spec"])

    def test_returns_all_keys(self):
        cm = np.array([[30, 10], [5, 55]])
        assert set(metrics_from_cm(cm)) == {"acc", "sens", "spec", "f1"}


class TestExtractMetrics:

    def test_output_keys(self, fake_artefacts):
        result = extract_metrics(fake_artefacts)
        assert set(result) == {"auc", "acc", "sens", "spec", "f1"}

    def test_list_lengths_match_input(self, fake_artefacts):
        result = extract_metrics(fake_artefacts)
        n = len(fake_artefacts)
        for key in result:
            assert len(result[key]) == n

    def test_values_in_range(self, fake_artefacts):
        result = extract_metrics(fake_artefacts)
        for key in result:
            for v in result[key]:
                assert 0.0 <= v <= 1.0, f"{key}={v} out of [0, 1]"
