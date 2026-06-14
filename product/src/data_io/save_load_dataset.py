"""
Utility functions for the ABIDE dataset.

This module provides functionality to load raw fMRI time-series data, 
extract and filter phenotypic metadata, and handle the persistent storage 
and retrieval of harmonised datasets.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from utils.paths import display_path, get_project_root
from sklearn.model_selection import train_test_split



def load_timeseries(folder_path: str) -> list[str]:
    """
    Loads all timeseries files with the .1D extension from a specified folder.

    Args:
        folder_path: Path to the folder containing timeseries .1D files.

    Returns:
        A sorted list of absolute file paths to all .1D timeseries files 
        found in the folder.
    """
    path = Path(folder_path)
    file_paths = sorted([str(p) for p in path.glob('*.1D')])
    print(f"Found {len(file_paths)} files")
    return file_paths


def load_phenotypic_data(
    pheno_path: str,
    columns: List[str] = None
) -> pd.DataFrame:
    """
    Loads phenotypic metadata from a CSV file.

    Args:
        pheno_path: Path to the phenotypic data CSV file.
        columns: List of column names to load. If None, defaults to a 
            predefined set of quality control and demographic features 
            (e.g., SITE_ID, DX_GROUP, AGE_AT_SCAN, func_mean_fd).

    Returns:
        A pandas DataFrame containing the selected phenotypic data.
    """
    if columns is None:
        columns = [
            'subject', 'SITE_ID', 'DX_GROUP', 'AGE_AT_SCAN', 'SEX',
            'FIQ', 'VIQ', 'PIQ', 'HANDEDNESS_CATEGORY', 'EYE_STATUS_AT_SCAN',
            'func_mean_fd', 'func_num_fd', 'func_perc_fd',
            'func_dvars', 'func_outlier',
            'func_efc', 'func_fber', 'func_fwhm',
            'func_quality', 'func_gsr'
        ]

    pheno = pd.read_csv(pheno_path, usecols=columns)
    print(f"Loaded phenotypic data: {len(pheno)} subjects")
    return pheno


def save_dataset(
    processed_dir: Path,
    X_harmonised: Optional[np.ndarray] = None,
    X_raw: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    metadata: Optional[pd.DataFrame] = None,
    feature_labels: Optional[pd.DataFrame] = None,
    coords: Optional[pd.DataFrame] = None,
    prefix: str = "abide",
) -> None:
    """
    Saves feature matrices and labels into a compressed 
    .npz file, while metadata and feature labels are saved as CSVs.

    Args:
        X_harmonised: Harmonised feature matrix (N subjects, D features).
        X_raw: Original, unharmonised feature matrix.
        y: Target labels (N subjects,).
        metadata: Phenotypic metadata associated with the samples.
        feature_labels: Labels for the ROI features (e.g., Atlas regions).
        processed_dir: Directory where the files will be saved.
        prefix: Filename prefix for the saved files. Defaults to "abide".
    """
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    if all(v is not None for v in [X_harmonised, X_raw, y, metadata]):
        prefix_dir = processed_dir / prefix
        prefix_dir.mkdir(parents=True, exist_ok=True)
        
        # Save arrays
        npz_path = prefix_dir / f"{prefix}_harmonised.npz"
        np.savez_compressed(
            npz_path,
            X=X_harmonised,
            X_raw=X_raw,
            y=y
        )
        print(f"Saved dataset to: {display_path(npz_path)}")

        # Save metadata
        metadata_path = prefix_dir / f"{prefix}_metadata.csv"
        metadata.to_csv(metadata_path, index=False)
        print(f"Saved metadata to: {display_path(metadata_path)}")

    if all(v is not None for v in [feature_labels, coords]):
        # Save feature labels
        feature_labels_path = processed_dir / f"feature_labels.csv"
        feature_labels.to_csv(feature_labels_path, index=False)
        print(f"Saved feature labels to: {display_path(feature_labels_path)}")
        
        # Save coordinates
        coords_path = processed_dir / f"coords.csv"
        coords.to_csv(coords_path, index=False)
        print(f"Saved coordinates to: {display_path(coords_path)}")


def load_dataset(
    prefix: str = "abide",
    verbose: bool = False,
) -> tuple:
    """
    Loads processed dataset arrays, metadata, and feature labels.

    Args:
        prefix: Filename prefix used during saving. Defaults to "abide".
        verbose: If True, prints shapes and label balance of the loaded data.

    Returns:
        A 5-tuple containing:
            - X_harmonised: Harmonised feature matrix.
            - X_raw: Original feature matrix.
            - y: Target labels.
            - metadata: Subject phenotypic metadata.
            - feature_labels: ROI feature labels.

    Raises:
        FileNotFoundError: If any of the required files (npz, metadata CSV, 
            or labels CSV) are missing from the processed directory.
    """ 

    root = get_project_root()
    processed_dir = root / "product" / "data" / "processed"
    prefix_dir = processed_dir / prefix

    # Load arrays
    npz_path = prefix_dir / f"{prefix}_harmonised.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)
    X_harmonised = data['X']
    X_raw = data['X_raw']
    y = data['y']

    # Load metadata
    metadata_path = prefix_dir / f"{prefix}_metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata = pd.read_csv(metadata_path)

    # Load feature labels
    feature_labels_path = processed_dir / f"feature_labels.csv"
    if not feature_labels_path.exists():
        raise FileNotFoundError(
            f"Feature labels not found: {feature_labels_path}")

    feature_labels = pd.read_csv(feature_labels_path)

    if verbose:
        print(f"Loaded dataset.")
        print(f"X shape : {X_harmonised.shape}")
        print(f"y shape : {y.shape}")
        print(f"ASD={int((y==1).sum())}  Control={int((y==0).sum())}")
        
    return X_harmonised, X_raw, y, metadata, feature_labels

def mixed_subset():
    """
    Creates a balanced mixed-sex subset from separate male and female datasets.

    Returns:
        A 4-tuple containing:
            - X: Concatenated and shuffled feature matrix.
            - y: Concatenated and shuffled target labels.
            - metadata: Aligned phenotypic metadata.
            - feature_labels: The labels/ROI pairs for the features.
    """

    # 1. Capture feature_labels
    X_m, _, y_m, meta_m, feature_labels = load_dataset("male")
    X_f, _, y_f, meta_f, _ = load_dataset("female")

    n_female = 81
    n_male = 82

    # Stratified sample from females
    X_f_mix, _, y_f_mix, _, meta_f_mix, _ = train_test_split(
        X_f, y_f, meta_f,
        train_size=n_female,
        stratify=y_f,
        random_state=42
    )

    # Stratified sample from males
    X_m_mix, _, y_m_mix, _, meta_m_mix, _ = train_test_split(
        X_m, y_m, meta_m,
        train_size=n_male,
        stratify=y_m,
        random_state=42
    )

    # Combine
    X_mixed = np.concatenate([X_f_mix, X_m_mix], axis=0)
    y_mixed = np.concatenate([y_f_mix, y_m_mix], axis=0)
    meta_mixed = pd.concat([meta_f_mix, meta_m_mix], axis=0)

    # Shuffle
    shuffle_idx = np.random.RandomState(seed=42).permutation(len(y_mixed))

    X = X_mixed[shuffle_idx]
    y = y_mixed[shuffle_idx]
    metadata = meta_mixed.iloc[shuffle_idx].reset_index(drop=True)

    print(f"Final Mixed Set: {X.shape[0]} participants")

    return X, y, metadata, feature_labels
