"""Tests for evaluation/metrics.py."""

import numpy as np
import pytest
import torch

from evaluation.metrics import aggregate_fold_metrics, calculate_metrics, ensure_numpy


class TestEnsureNumpy:

    def test_tensor_input(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        result = ensure_numpy(t)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_numpy_passthrough(self):
        a = np.array([1.0, 2.0])
        result = ensure_numpy(a)
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, a)

    def test_list_input(self):
        result = ensure_numpy([0, 1, 0])
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [0, 1, 0])


class TestCalculateMetrics:

    @pytest.fixture()
    def perfect(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        preds  = np.array([0, 0, 0, 1, 1, 1])
        probs  = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
        return preds, probs, labels

    @pytest.fixture()
    def all_wrong(self):
        labels = np.array([0, 0, 0, 1, 1, 1])
        preds  = np.array([1, 1, 1, 0, 0, 0])
        probs  = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
        return preds, probs, labels

    def test_perfect_predictions(self, perfect):
        scalars, _ = calculate_metrics(*perfect)
        assert scalars['acc']  == pytest.approx(1.0)
        assert scalars['sens'] == pytest.approx(1.0)
        assert scalars['spec'] == pytest.approx(1.0)
        assert scalars['f1']   == pytest.approx(1.0)
        assert scalars['auc']  == pytest.approx(1.0)

    def test_all_wrong_predictions(self, all_wrong):
        scalars, _ = calculate_metrics(*all_wrong)
        assert scalars['acc']  == pytest.approx(0.0)
        assert scalars['sens'] == pytest.approx(0.0)
        assert scalars['spec'] == pytest.approx(0.0)

    def test_returns_scalar_keys(self, perfect):
        scalars, _ = calculate_metrics(*perfect)
        assert set(scalars) == {'acc', 'auc', 'f1', 'sens', 'spec'}

    def test_returns_artefact_keys(self, perfect):
        _, artefacts = calculate_metrics(*perfect)
        assert set(artefacts) == {'roc', 'cm'}

    def test_cm_layout(self, perfect):
        labels = np.array([0, 0, 1, 1])
        preds  = np.array([0, 1, 0, 1])   # TN=1, FP=1, FN=1, TP=1
        probs  = np.array([0.1, 0.9, 0.1, 0.9])
        _, artefacts = calculate_metrics(preds, probs, labels)
        cm = artefacts['cm']
        assert cm[0, 0] == 1  # TN
        assert cm[0, 1] == 1  # FP
        assert cm[1, 0] == 1  # FN
        assert cm[1, 1] == 1  # TP

    def test_accepts_tensors(self, perfect):
        preds, probs, labels = perfect
        scalars_np, _ = calculate_metrics(preds, probs, labels)
        scalars_t, _  = calculate_metrics(
            torch.tensor(preds, dtype=torch.float32),
            torch.tensor(probs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
        for key in scalars_np:
            assert scalars_np[key] == pytest.approx(scalars_t[key])


class TestAggregateFoldMetrics:

    @pytest.fixture()
    def two_folds(self):
        return [
            {'acc': 0.8, 'auc': 0.9, 'f1': 0.7, 'sens': 0.75, 'spec': 0.85},
            {'acc': 0.6, 'auc': 0.7, 'f1': 0.5, 'sens': 0.55, 'spec': 0.65},
        ]

    def test_returns_all_keys(self, two_folds):
        result = aggregate_fold_metrics(two_folds)
        expected = {'acc', 'auc', 'f1', 'sens', 'spec',
                    'std_acc', 'std_auc', 'std_f1', 'std_sens', 'std_spec'}
        assert set(result) == expected

    def test_mean_values(self, two_folds):
        result = aggregate_fold_metrics(two_folds)
        assert result['acc']  == pytest.approx(0.7)
        assert result['auc']  == pytest.approx(0.8)
        assert result['sens'] == pytest.approx(0.65)

    def test_std_values(self, two_folds):
        result = aggregate_fold_metrics(two_folds)
        assert result['std_acc'] == pytest.approx(np.std([0.8, 0.6]))
        assert result['std_auc'] == pytest.approx(np.std([0.9, 0.7]))

    def test_single_fold_std_is_zero(self):
        result = aggregate_fold_metrics(
            [{'acc': 0.8, 'auc': 0.9, 'f1': 0.7, 'sens': 0.75, 'spec': 0.85}]
        )
        for key in ['std_acc', 'std_auc', 'std_f1', 'std_sens', 'std_spec']:
            assert result[key] == pytest.approx(0.0)
