"""
Timeseries Preprocessing Module.

This module provides utilities for cleaning and filtering multi-dimensional 
timeseries data, specifically designed for data structured as (timepoints, ROIs). 

The primary workflow involves:
    1. Validating signal integrity (removing NaNs and Infs).
    2. Enforcing minimum temporal duration requirements.
    3. Filtering or neutralising Regions of Interest (ROIs) with low variance.
    4. Batch processing multiple files while tracking success/failure rates.

"""

from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np


def preprocess_timeseries(
    ts: np.ndarray,
    min_timepoints: int = 50,
    max_bad_roi_ratio: float = 0.3
) -> Optional[np.ndarray]:
    """
    Preprocess a timeseries matrix by removing invalid timepoints and handling bad ROIs.

    Steps:
        1. Removes timepoints containing NaN or infinite values.
        2. Checks that the remaining number of timepoints meets min_timepoints.
        3. Identifies ROIs with non-finite or near-zero variance.
        4. If the fraction of bad ROIs exceeds max_bad_roi_ratio, the timeseries is discarded.
        5. Replaces bad ROIs with zeros in the cleaned timeseries.

    Args:
        ts (np.ndarray): Timeseries array of shape (n_timepoints, n_rois).
        min_timepoints (int, optional): Minimum number of valid timepoints required to keep the 
        timeseries. Defaults to 50.
        max_bad_roi_ratio (float, optional): Maximum allowed fraction of bad ROIs. Defaults to 0.3.

    Returns:
        ts_clean (Optional[np.ndarray]): The cleaned timeseries array of shape (n_valid_timepoints, 
        n_rois).
    """
    # Drop timepoints with NaN/inf
    valid_timepoints = np.isfinite(ts).all(axis=1)
    ts_clean = ts[valid_timepoints]

    if len(ts_clean) < min_timepoints:
        return None

    # Identify ROIs with near-zero or non-finite variance
    roi_variance = np.var(ts_clean, axis=0)
    bad_rois = ~np.isfinite(roi_variance) | (roi_variance < 1e-10)

    # Check if too many bad ROIs
    bad_roi_fraction = np.mean(bad_rois)
    if bad_roi_fraction > max_bad_roi_ratio:
        return None

    # Replace bad ROIs with zeros
    ts_clean[:, bad_rois] = 0

    return ts_clean


def preprocess_all(
    file_paths: List[str]
) -> Tuple[List[str], List[np.ndarray]]:
    """
    Preprocess multiple timeseries files, filtering out invalid or low-quality data.

    Iterates over a list of file paths, loads each timeseries, and applies
    preprocess_timeseries to clean the data. Timeseries that fail quality
    checks (too few timepoints or too many bad ROIs) are skipped.

    Args:
        file_paths (List[str]): List of file paths to timeseries files.

    Returns:
        tuple: A tuple containing:
            - valid_file_paths (List[str]): File paths corresponding to timeseries that passed 
            preprocessing.
            - cleaned_time_series (List[np.ndarray]): Cleaned timeseries arrays corresponding to 
            the valid files.
    """
    cleaned_time_series = []
    valid_file_paths = []
    skipped = []

    for file_path in file_paths:
        ts = np.genfromtxt(file_path)
        ts_clean = preprocess_timeseries(ts)

        if ts_clean is not None:
            cleaned_time_series.append(ts_clean)
            valid_file_paths.append(file_path)
        else:
            skipped.append(Path(file_path).name)

    print(f"Processed {len(cleaned_time_series)} subjects, skipped {len(skipped)}")
    if skipped:
        print("Skipped files:")
        for fname in skipped:
            print(f"  {fname}")
    return valid_file_paths, cleaned_time_series
