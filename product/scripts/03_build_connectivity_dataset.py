"""
Prepare connectivity-based dataset from raw fMRI time series.
Filter already downloaded files.

Usage:
    python 03_build_connectivity_dataset.py --prefix NYU
    python 03_build_connectivity_dataset.py --prefix female --sex 2
    python 03_build_connectivity_dataset.py --prefix male_matched --match-to-female-metadata data/
    processed/female_metadata.csv
"""

import argparse
from pathlib import Path

import numpy as np

from utils.paths import get_project_root

from data_io.save_load_dataset import (
    load_timeseries,
    load_phenotypic_data,
    save_dataset,
)
from preprocessing.timeseries import preprocess_all
from preprocessing.participants import (
    extract_local_ids,
    match_participants,
    filter_phenotypic_data,
    match_males_to_females,
)
from preprocessing.connectivity import compute_connectivity_matrices, generate_feature_labels
from preprocessing.harmonisation import apply_harmonisation
from preprocessing.visualisation import print_data_summary

import pandas as pd


def main(
    raw_dir: str,
    pheno_path: str,
    processed_dir: str,
    prefix: str,
    sex: int | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
    female_metadata_path: Path | None = None,
    age_tolerance: float = 2.0,
    seed: int = 42,
):
    """
    Build connectivity dataset from raw fMRI time series.

    Pipeline:
    1. Load raw time series and phenotypic data
    2. Optionally filter phenotypic data by subgroup (sex, age) OR match males to a saved female 
    metadata set
    3. Preprocess time series (filtering, normalisation)
    4. Compute functional connectivity matrices
    5. Match with phenotypic labels
    6. Apply ComBat harmonisation for multi-site data
    7. Save processed dataset

    Args:
        raw_dir (str): Directory containing raw .1D time series files.
        pheno_path (str): Path to phenotypic CSV file.
        output_dir (str): Directory to save processed dataset.
        roi_labels_path (str): Path to CC200_ROI_labels.csv for anatomical labels.
        prefix (str): Filename prefix for the saved files.
        sex (int): Biological sex filter. 1 for Male, 2 for Female. If None, includes both.
        age_min (float): Minimum age threshold (inclusive). If None, no lower bound.
        age_max (float): Maximum age threshold (inclusive). If None, no upper bound.
        female_metadata_path (Path): Path to a saved female_metadata.csv. If provided,
            matches males to those females by site and age instead of applying
            a simple sex filter. Mutually exclusive with --sex.
        age_tolerance (float): Maximum age difference in years when matching males to
            females. Only used when female_metadata_path is provided.
        seed (int): Random seed for reproducibility of male matching.
    """

    # Load raw data
    print("\nLoading Data...")
    file_paths = load_timeseries(raw_dir)
    pheno = load_phenotypic_data(pheno_path)

    if female_metadata_path is not None:
        # Match males to the exact female set that survived preprocessing
        print("\nMatching males to female subset...")
        female_metadata = pd.read_csv(female_metadata_path)
        female_ids = set(
            female_metadata["ID"].str.split("_").str[-1].astype(int)
        )
        print(f"  Female participants from prior run: {len(female_ids)}")

        # Preprocess first to identify which subjects have valid files
        print("\nPreprocessing Time Series...")
        valid_paths, time_series_list = preprocess_all(file_paths)
        local_ids = extract_local_ids(valid_paths)
        available_ids = {int(id_.split("_")[-1]) for id_ in local_ids}

        pheno, _ = match_males_to_females(
            pheno,
            female_ids=female_ids,
            age_tolerance=age_tolerance,
            seed=seed,
            available_ids=available_ids,
        )

    else:
        # Simple subgroup filter
        if any(f is not None for f in [sex, age_min, age_max]):
            print("\nFiltering Participants...")
            pheno = filter_phenotypic_data(pheno, sex=sex, age_min=age_min, age_max=age_max)

        # Preprocess time series
        print("\nPreprocessing Time Series...")
        valid_paths, time_series_list = preprocess_all(file_paths)
        local_ids = extract_local_ids(valid_paths)

    # Compute functional connectivity
    print("\nComputing Connectivity Matrices...")
    # Connectivity matrices become the input features for classification
    X = compute_connectivity_matrices(time_series_list)

    # Match connectivity data with phenotypic labels
    print("\nMatching participants...")
    X, metadata, _ = match_participants(local_ids, pheno, X)
    y = metadata['DX_GROUP'].values
    
    # Impute FIQ: replace sentinel values, then fill by site median then global median
    metadata["FIQ"] = metadata["FIQ"].replace(-9999.0, np.nan)
    print(
        f"FIQ missing after replacing sentinels: {metadata['FIQ'].isna().sum()}")
    metadata["FIQ"] = metadata.groupby("SITE_ID")["FIQ"].transform(
        lambda x: x.fillna(x.median())
    )
    metadata["FIQ"] = metadata["FIQ"].fillna(metadata["FIQ"].median())
    print(f"FIQ missing after imputation: {metadata['FIQ'].isna().sum()}")

    # Harmonise across imaging sites
    print("\nApplying Harmonisation...")
    X_raw = X.copy()
    X_harmonised = apply_harmonisation(X, metadata)

    # Save complete dataset
    print("\nSaving Dataset...")
    processed_dir = Path(processed_dir)
    save_dataset(processed_dir=processed_dir, X_harmonised=X_harmonised, X_raw=X_raw, y=y, metadata=metadata, prefix=prefix)

    # Print summary statistics
    print_data_summary(X_harmonised, y, metadata)


if __name__ == "__main__":

    ROOT = get_project_root()
    DATA = ROOT / "product" / "data"
    PHENO = DATA / "external" / "Phenotypic_V1_0b_preprocessed.csv"
    RAW = DATA / "raw"
    PROC = DATA / "processed"

    p = argparse.ArgumentParser(
        description="Prepare connectivity-based dataset from raw fMRI time series.")
    p.add_argument("--prefix", type=str, default=None,
                help="Filename prefix for the saved files.")
    p.add_argument("--sex", type=int, default=None, choices=[1, 2],
                help="Sex filter: 1 = Male, 2 = Female.")
    p.add_argument("--age-min", type=float, default=None,
                help="Minimum age (inclusive).")
    p.add_argument("--age-max", type=float, default=None,
                help="Maximum age (inclusive).")
    p.add_argument("--match-to-female-metadata", type=Path, default=None,
                help="Path to female_metadata.csv. If provided, matches males "
                    "to that female set by site and age. Cannot be used with --sex.")
    p.add_argument("--age-tolerance", type=float, default=2.0,
                help="Maximum age difference in years when matching males to "
                    "females (default: 2.0). Only used with --match-to-female-metadata.")
    p.add_argument("--seed", type=int, default=42,
                help="Random seed for male matching reproducibility (default: 42).")
    p.add_argument("--subdir", type=str, default='abide',
                help="Subdirectory where raw files are stored.")
    args = p.parse_args()

    if args.match_to_female_metadata and args.sex:
        p.error("--match-to-female-metadata and --sex are mutually exclusive.")

    female_metadata_path = (
        ROOT / args.match_to_female_metadata
        if args.match_to_female_metadata is not None
        else None
    )

    subdir = (
        RAW / args.subdir
        if args.subdir is not None
        else RAW
    )

    main(subdir, PHENO, PROC, args.prefix,
            sex=args.sex,
            age_min=args.age_min,
            age_max=args.age_max,
            female_metadata_path=female_metadata_path,
            age_tolerance=args.age_tolerance,
            seed=args.seed
        )
