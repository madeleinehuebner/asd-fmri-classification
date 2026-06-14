"""
Centralised logging and artefact management for experiments.

This module provides a unified interface for saving scalar experimental results
to a CSV log and persisting artefacts to compressed NumPy files.
"""

from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from utils.paths import display_path, get_project_root


def _to_json_serialisable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serialisable(v) for v in obj]
    if torch.is_tensor(obj):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return repr(obj)


COLUMN_ORDER: list[str] = [
    "timestamp",
    "meta_site",
    "meta_sex",
    "meta_n_participants",
    "meta_n_asd",
    "meta_n_td",
    "meta_eval_strategy",
    "meta_n_folds",
    "metric_acc",
    "metric_auc",
    "metric_f1",
    "metric_sens",
    "metric_spec",
    "metric_std_acc",
    "metric_std_auc",
    "metric_std_f1",
    "metric_std_sens",
    "metric_std_spec",
    "cfg_notes",
]


class CentralisedLogger:
    """
    Saves experiment results into a central CSV and artefacts in subdirectories.

    Attributes:
        root: The path to the project root.
        results_dir: The directory where results are stored.
        artefact_path: The base directory for model-specific artefacts.
        csv_path: The path to the summary CSV log file.
    """

    def __init__(
        self,
        model_name: str,
        custom_root: Optional[Path] = None
    ) -> None:
        """
        Initialises logger and creates directory structures.

        Args:
            model_name: Name of the model being evaluated. Used for naming
                the CSV and the artefact subfolder.
            custom_root: Override for the project root path.
        """
        self.root: Path = custom_root or get_project_root()

        self.results_dir: Path = self.root / "product" / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.artefact_path: Path = self.results_dir / "artefacts" / model_name / "outputs"
        self.artefact_path.mkdir(parents=True, exist_ok=True)

        self.csv_path: Path = self.results_dir / f"{model_name}_experiment_log.csv"
        
        self._last_run_dir: Optional[Path] = None

        # Accumulates one timestamp per save_artefacts call within a run.
        # The last entry becomes the folder name and the CSV timestamp.
        self._fold_timestamps: List[str] = []

    def _align_to_existing_columns(self, log_entry: Dict[str, Any]) -> pd.DataFrame:
        # Build ordered row from COLUMN_ORDER
        ordered_data = {col: log_entry.get(col, None) for col in COLUMN_ORDER}

        for k, v in log_entry.items():
            if k not in ordered_data:
                ordered_data[k] = v

        return pd.DataFrame([ordered_data])

    def log_run(
        self,
        meta: Dict[str, Any],
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        search_space: Optional[Dict[str, Any]] = None,
        best_params: Optional[Dict[str, Any]] = None,
        additional_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Assembles and writes a scalar log entry to the central CSV, and saves
        params to a JSON file in the artefacts run directory.

        Args:
            meta: Dictionary of metadata.
            params: Dictionary of hyperparameters.
                Non-JSON-serialisable values are stored as repr() strings.
            metrics: Dictionary of result scalars.
            search_space: Optional hyperparameter search space dict.
            best_params: Optional best params from hyperparameter tuning.
            additional_config: Optional dictionary for extra configuration notes.
        """
        if self._fold_timestamps:
            run_timestamp = self._fold_timestamps[-1]
        else:
            run_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Save params as JSON in the artefacts run directory
        run_dir = self.artefact_path / run_timestamp.replace(':', '-').replace(' ', '_')
        run_dir.mkdir(parents=True, exist_ok=True)

        params_payload: Dict[str, Any] = {"pipeline_params": _to_json_serialisable(params)}
        if search_space is not None:
            params_payload["search_space"] = _to_json_serialisable(search_space)
        if best_params is not None:
            params_payload["best_params"] = _to_json_serialisable(best_params)

        with open(run_dir / "params.json", "w") as f:
            json.dump(params_payload, f, indent=2)
        print(f"Params saved to: {display_path(run_dir / 'params.json')}")

        # Filter and prefix metrics
        processed_metrics: Dict[str, Union[int, float]] = {}
        for k, v in metrics.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    processed_metrics[f"metric_{k}"] = v.item()
                else:
                    continue  # Skip multi-element tensors
            elif isinstance(v, (tuple, list)):
                continue  # Skip curve outputs
            elif isinstance(v, (int, float, np.number)):
                processed_metrics[f"metric_{k}"] = v

        log_entry: Dict[str, Any] = {
            'timestamp': run_timestamp,
            **{f"meta_{k}": v for k, v in meta.items()},
            **processed_metrics,
        }

        if additional_config:
            log_entry.update({f"cfg_{k}": v for k, v in additional_config.items()})

        aligned_row = self._align_to_existing_columns(log_entry)

        needs_header = (
            not self.csv_path.exists()
            or self.csv_path.stat().st_size == 0
        )

        if needs_header:
            all_cols = list(aligned_row.columns)
            for col in COLUMN_ORDER:
                if col not in all_cols:
                    all_cols.append(col)
            aligned_row = aligned_row.reindex(columns=all_cols)

        aligned_row.to_csv(
            self.csv_path, mode='a', index=False,
            header=needs_header
        )
        print(f"Summary logged to: {display_path(self.csv_path)}")
        self._fold_timestamps = []

    def save_artefacts(
        self,
        results: Dict[str, Any],
        prefix: str = "run",
        verbose: bool = True
    ) -> None:
        """     
        Saves detailed model outputs to a compressed .npz file.

        Files are stored in a subdirectory named after the most recent fold's
        timestamp. If multiple folds are saved sequentially, the directory is
        renamed on each call to reflect the latest timestamp, ensuring all 
        artefacts for a single run stay grouped.

        Args:
            results: A dictionary containing:
                - 'roc': Tuple/List of (fpr, tpr, thresholds) as Tensors.
                - 'cm': Confusion matrix as a Tensor.
            prefix: String prefix for the filename (e.g., "fold1").
            verbose: If True, prints the final save path.
        """
        this_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._fold_timestamps.append(this_timestamp)

        new_dir_name = this_timestamp.replace(':', '-').replace(' ', '_')
        new_run_dir  = self.artefact_path / new_dir_name

        if len(self._fold_timestamps) > 1:
            # Rename the directory from the previous fold's timestamp to this one
            prev_timestamp = self._fold_timestamps[-2]
            prev_dir_name  = prev_timestamp.replace(':', '-').replace(' ', '_')
            prev_run_dir   = self.artefact_path / prev_dir_name
            if prev_run_dir.exists():
                prev_run_dir.rename(new_run_dir)

        new_run_dir.mkdir(parents=True, exist_ok=True)

        # File is named with this fold's own timestamp for individual traceability
        this_dir_name = this_timestamp.replace(':', '-').replace(' ', '_')
        file_name  = f"{prefix}_{this_dir_name}.npz"
        save_path  = new_run_dir / file_name

        fpr, tpr, thresholds = [
            t.detach().cpu().numpy() if torch.is_tensor(t) else t
            for t in results['roc']
        ]
        cm = (
            results['cm'].detach().cpu().numpy()
            if torch.is_tensor(results['cm'])
            else results['cm']
        )

        np.savez_compressed(save_path, fpr=fpr, tpr=tpr, thresholds=thresholds, cm=cm)
        if verbose:
            print(f"Detailed artefacts saved to: {display_path(save_path)}")

    def load_artefact(
        self,
        file_name: str,
        folder_path: Optional[Path] = None
    ) -> Dict[str, np.ndarray]:
        """
        Loads a single .npz artefact file.

        Args:
            file_name: Name of the .npz file to load.
            folder_path: Directory to search. Defaults to self.artefact_path.

        Returns:
            Dictionary mapping array names ('fpr', 'tpr', etc.) to NumPy arrays.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        load_dir = folder_path or self.artefact_path
        full_path = load_dir / file_name

        if not full_path.exists():
            raise FileNotFoundError(f"No artefact found at {full_path}")

        with np.load(full_path) as data:
            return {key: data[key] for key in data.files}

    def load_folder_artefacts(
        self,
        folder_path: Optional[Path] = None,
        pattern: str = "*.npz"
    ) -> Dict[str, Dict]:
        """
        Loads all .npz files matching a pattern in a directory.

        Args:
            folder_path: Directory to search. Defaults to self.artefact_path.
            pattern: Glob pattern for matching files.

        Returns:
            Dictionary mapping filenames to their loaded array contents.
        """
        target_dir = folder_path or self.artefact_path
        return {
            file_path.name: self.load_artefact(file_path.name, folder_path=target_dir)
            for file_path in target_dir.glob(pattern)
        }
        
    def prepare_meta(
        self, 
        metadata: pd.DataFrame, 
        y: Union[np.ndarray, torch.Tensor, pd.Series],
        eval_strategy: str = "Leave-One-Site-Out",
        n_folds: int = 5,
    ) -> Dict[str, Any]:
        """
        Automatically generates the meta_info dictionary from experimental data.
        """
        site_id_col = 'SITE_ID' if 'SITE_ID' in metadata.columns else 'site'
        unique_sites = metadata[site_id_col].unique()
        site_label = unique_sites[0] if len(unique_sites) == 1 else "ALL"

        sex_col = 'SEX' if 'SEX' in metadata.columns else 'sex'
        n_sexes = metadata[sex_col].nunique()
        if n_sexes > 1:
            sex_label = "Mixed"
        else:
            val = metadata[sex_col].iloc[0]
            sex_label = "Male" if val == 1 else "Female"

        # Count Logic
        if torch.is_tensor(y):
            n_asd = int((y == 1).sum().item())
            n_td  = int((y == 0).sum().item())
            total = len(y)
        else:
            n_asd = int((y == 1).sum())
            n_td  = int((y == 0).sum())
            total = len(y)

        return {
            'site': site_label,
            'sex': sex_label,
            'n_participants': total,
            'n_asd': n_asd,
            'n_td': n_td,
            'eval_strategy': eval_strategy,
            'n_folds': n_folds,
        }
    @property
    def current_run_dir(self) -> Optional[Path]:
        """Returns the path to the current or most recent run directory."""
        if self._fold_timestamps:
            latest_ts = self._fold_timestamps[-1]
            new_dir_name = latest_ts.replace(':', '-').replace(' ', '_')
            self._last_run_dir = self.artefact_path / new_dir_name
        
        return self._last_run_dir