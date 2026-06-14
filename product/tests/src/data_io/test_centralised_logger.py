"""Tests for data_io/centralised_logger.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from data_io.centralised_logger import CentralisedLogger, COLUMN_ORDER


def _make_logger(tmp_path, model_name="test_model"):
    return CentralisedLogger(model_name, custom_root=tmp_path)


class TestInit:

    def test_directories_created_and_csv_absent(self, tmp_path):
        """results_dir and artefact_path are created on init; CSV is not yet written."""
        logger = _make_logger(tmp_path, model_name="my_gcn")
        assert logger.results_dir.exists() and logger.results_dir.is_dir()
        assert logger.artefact_path.exists() and "my_gcn" in str(logger.artefact_path)
        assert logger.csv_path.name == "my_gcn_experiment_log.csv"
        assert not logger.csv_path.exists()
        assert logger._fold_timestamps == []


class TestAlignToExistingColumns:

    def test_returns_single_row_with_canonical_columns(self, tmp_path):
        logger = _make_logger(tmp_path)
        row = logger._align_to_existing_columns({"metric_acc": 0.9})
        assert isinstance(row, pd.DataFrame) and len(row) == 1
        assert "metric_acc" in row.columns

    def test_missing_canonical_columns_are_none(self, tmp_path):
        logger = _make_logger(tmp_path)
        row = logger._align_to_existing_columns({"timestamp": "t"})
        assert "metric_acc" in row.columns and pd.isna(row["metric_acc"].iloc[0])


class TestLogRun:

    def test_csv_created_and_appended(self, tmp_path, sample_meta, sample_params, sample_metrics):
        logger = _make_logger(tmp_path)
        logger.log_run(sample_meta, sample_params, sample_metrics)
        assert logger.csv_path.exists()
        assert len(pd.read_csv(logger.csv_path)) == 1
        logger.log_run(sample_meta, sample_params, sample_metrics)
        assert len(pd.read_csv(logger.csv_path)) == 2

    def test_scalar_metric_and_meta_logged(self, tmp_path, sample_meta, sample_params, sample_metrics):
        logger = _make_logger(tmp_path)
        logger.log_run(sample_meta, sample_params, sample_metrics)
        df = pd.read_csv(logger.csv_path)
        assert pytest.approx(df["metric_acc"].iloc[0]) == sample_metrics["acc"]
        assert "meta_site" in df.columns and df["meta_site"].iloc[0] == sample_meta["site"]

    def test_tensor_and_list_metrics_filtered(self, tmp_path, sample_meta, sample_params):
        """Single-element tensors are logged; multi-element tensors and lists are excluded."""
        logger = _make_logger(tmp_path)
        metrics = {
            "acc":   torch.tensor(0.85),
            "roc":   torch.tensor([0.0, 0.5, 1.0]),
            "curve": [0.1, 0.5, 0.9],
        }
        logger.log_run(sample_meta, sample_params, metrics)
        df = pd.read_csv(logger.csv_path)
        assert pytest.approx(df["metric_acc"].iloc[0], abs=1e-4) == 0.85
        assert "metric_roc" not in df.columns
        assert "metric_curve" not in df.columns

    def test_list_param_excluded(self, tmp_path, sample_meta, sample_metrics):
        logger = _make_logger(tmp_path)
        logger.log_run(sample_meta, {"lr": 0.001, "hidden_sizes": [64, 32]}, sample_metrics)
        df = pd.read_csv(logger.csv_path)
        assert "params_hidden_sizes" not in df.columns

    def test_additional_config_logged(self, tmp_path, sample_meta, sample_params, sample_metrics):
        logger = _make_logger(tmp_path)
        logger.log_run(sample_meta, sample_params, sample_metrics,
                       additional_config={"notes": "baseline run"})
        df = pd.read_csv(logger.csv_path)
        assert df["cfg_notes"].iloc[0] == "baseline run"

    def test_fold_timestamp_used_and_cleared(self, tmp_path, sample_meta, sample_params, sample_metrics):
        logger = _make_logger(tmp_path)
        logger._fold_timestamps = ["2020-01-01 12:00:00"]
        logger.log_run(sample_meta, sample_params, sample_metrics)
        df = pd.read_csv(logger.csv_path)
        assert df["timestamp"].iloc[0] == "2020-01-01 12:00:00"
        assert logger._fold_timestamps == []


class TestSaveArtefacts:

    def test_creates_npz_with_correct_keys_and_values(self, tmp_path, sample_artefacts):
        logger = _make_logger(tmp_path)
        logger.save_artefacts(sample_artefacts, prefix="fold1")
        npz_files = list(logger.artefact_path.rglob("*.npz"))
        assert len(npz_files) == 1
        assert "fold1" in npz_files[0].name
        data = np.load(npz_files[0])
        assert set(data.files) == {"fpr", "tpr", "thresholds", "cm"}
        np.testing.assert_array_equal(data["fpr"], sample_artefacts["roc"][0])
        np.testing.assert_array_equal(data["cm"], sample_artefacts["cm"])

    def test_multiple_folds_accumulate_files(self, tmp_path, sample_artefacts):
        logger = _make_logger(tmp_path)
        logger.save_artefacts(sample_artefacts, prefix="fold1")
        logger.save_artefacts(sample_artefacts, prefix="fold2")
        assert len(list(logger.artefact_path.rglob("*.npz"))) == 2

    def test_torch_tensor_artefacts_accepted(self, tmp_path):
        logger = _make_logger(tmp_path)
        artefacts = {
            'roc': (torch.tensor([0.0, 0.5, 1.0]),
                    torch.tensor([0.0, 0.7, 1.0]),
                    torch.tensor([1.0, 0.5, 0.0])),
            'cm': torch.tensor([[40, 10], [12, 38]]),
        }
        logger.save_artefacts(artefacts, prefix="fold1")
        assert len(list(logger.artefact_path.rglob("*.npz"))) == 1

    def test_fold_timestamp_appended(self, tmp_path, sample_artefacts):
        logger = _make_logger(tmp_path)
        assert len(logger._fold_timestamps) == 0
        logger.save_artefacts(sample_artefacts, prefix="fold1")
        assert len(logger._fold_timestamps) == 1


class TestLoadArtefact:

    def test_returns_dict_with_correct_keys_and_types(self, tmp_path, sample_artefacts):
        logger = _make_logger(tmp_path)
        logger.save_artefacts(sample_artefacts, prefix="fold1")
        npz_path = next(logger.artefact_path.rglob("*.npz"))
        result = logger.load_artefact(npz_path.name, folder_path=npz_path.parent)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"fpr", "tpr", "thresholds", "cm"}
        for v in result.values():
            assert isinstance(v, np.ndarray)

    def test_raises_for_missing_file(self, tmp_path):
        logger = _make_logger(tmp_path)
        with pytest.raises(FileNotFoundError):
            logger.load_artefact("nonexistent.npz")


class TestLoadFolderArtefacts:

    def test_loads_all_npz_and_returns_correct_structure(self, tmp_path, sample_artefacts):
        logger = _make_logger(tmp_path)
        logger.save_artefacts(sample_artefacts, prefix="fold1")
        logger.save_artefacts(sample_artefacts, prefix="fold2")
        npz_files = list(logger.artefact_path.rglob("*.npz"))
        folder = npz_files[0].parent
        result = logger.load_folder_artefacts(folder_path=folder)
        assert len(result) == len(npz_files)
        assert all(k.endswith(".npz") for k in result)
        for v in result.values():
            assert isinstance(v, dict)

    def test_empty_folder_returns_empty_dict(self, tmp_path):
        logger = _make_logger(tmp_path)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        assert logger.load_folder_artefacts(folder_path=empty_dir) == {}
