"""
Participant Data Mapping and Normalisation Module.

This module provides tools for matching neuroimaging timeseries data with 
phenotypic metadata. It handles the parsing of file-based identifiers, site 
name standardisation, and the alignment of feature matrices with participant 
records (e.g., diagnosis, age, and sex).

Key Capabilities:
    - Site Normalisation: Maps inconsistent site strings (e.g., 'MAXMUN_A') to a unified canonical 
    format ('MAX_MUN').
    - ID Parsing: Extracts site and participant components from complex local filenames.
    - Data Alignment: Matches physical file paths to phenotypic database rows, filtering out 
    participants missing either data component.
    - Metadata Preparation: Reformats diagnosis codes (Binary: 1=ASD, 0=Control) and extracts key 
    covariates for downstream analysis.
"""

from typing import List, Tuple, Dict
import os
import numpy as np
import pandas as pd


SITE_NORMALISATIONS = {
    # CMU subsections (letter suffixes)
    'CMU_A': 'CMU',
    'CMU_B': 'CMU',
    'CMU': 'CMU',

    # MAX_MUN subsections (letter suffixes + name change)
    'MAXMUN_A': 'MAX_MUN',
    'MAXMUN_B': 'MAX_MUN',
    'MAXMUN_C': 'MAX_MUN',
    'MAXMUN_D': 'MAX_MUN',
    'MAXMUN': 'MAX_MUN',
    'MAX_MUN': 'MAX_MUN',

    # Sites with numeric suffixes - preserve them for matching
    'LEUVEN_1': 'LEUVEN_1',
    'LEUVEN_2': 'LEUVEN_2',
    'UCLA_1': 'UCLA_1',
    'UCLA_2': 'UCLA_2',
    'UM_1': 'UM_1',
    'UM_2': 'UM_2',

    # All other sites (direct mapping)
    'CALTECH': 'CALTECH',
    'KKI': 'KKI',
    'NYU': 'NYU',
    'OHSU': 'OHSU',
    'OLIN': 'OLIN',
    'PITT': 'PITT',
    'SBL': 'SBL',
    'SDSU': 'SDSU',
    'STANFORD': 'STANFORD',
    'TRINITY': 'TRINITY',
    'USM': 'USM',
    'YALE': 'YALE',
}


def normalise_site(site: str) -> str:
    """
    Normalise site identifiers to a consistent format.

    This function attempts to standardise site labels by checking a predefined
    mapping SITE_NORMALISATIONS. If no direct mapping exists, it removes
    single-letter subsections (e.g., 'CMU_A' -> 'CMU', 'MAXMUN_B' -> 'MAX_MUN').

    Note: Numeric suffixes (e.g., _1, _2) are not removed here - they are 
    preserved for matching. Use remove_numeric_suffix() for display purposes.

    Args:
        site (str): Original site identifier string.

    Returns:
        site (str): Normalised site identifier according to SITE_NORMALISATIONS or a simplified version of 
        the original string.
    """
    # Check direct mapping first
    if site in SITE_NORMALISATIONS:
        return SITE_NORMALISATIONS[site]

    # Try to remove letter suffixes and check mapping again
    parts = site.split('_')
    if len(parts) >= 2:
        last_part = parts[-1]

        # Remove single-letter subsections (e.g., 'CMU_A' -> 'CMU')
        if last_part.isalpha() and len(last_part) == 1:
            base = '_'.join(parts[:-1])
            if base in SITE_NORMALISATIONS:
                return SITE_NORMALISATIONS[base]
            return base

    return site


def remove_numeric_suffix(site: str) -> str:
    """
    Remove numeric suffixes from site names for display purposes.

    Examples:
        'LEUVEN_1' -> 'LEUVEN'
        'UCLA_2' -> 'UCLA'
        'UM_1' -> 'UM'
        'CMU' -> 'CMU' (no change)

    Args:
        site (str): Site identifier string.

    Returns:
        site (str): Site name with numeric suffix removed.
    """
    parts = site.split('_')
    if len(parts) >= 2 and parts[-1].isdigit():
        return '_'.join(parts[:-1])
    return site


def parse_local_id(local_id: str) -> Tuple[str, str]:
    """
    Parse a local identifier into its site and participant components.

    Args:
        local_id (str): Local identifier string in the format "SITE_participant" (e.g., 
        "NYU_0050952").

    Returns:
        tuple: A tuple containing:
            - site (str): Uppercased site identifier extracted from the local ID.
            - participant (str): participant ID extracted from the local ID.
    """
    parts = local_id.split('_')
    participant = parts[-1]
    site = '_'.join(parts[:-1])
    return site.upper(), participant


def normalise_participant_id(participant_id: str) -> str:
    """
    Normalise a participant ID by removing leading zeros.

    If the input consists entirely of zeros, returns "0".

    Args:
        participant_id (str): Original participant identifier (e.g., "0050952").

    Returns:
        str: Normalised participant ID with leading zeros removed (e.g., "50952").
    """
    return participant_id.lstrip('0') or '0'


def extract_local_ids(file_paths: List[str]) -> List[str]:
    """
    Extract local participant identifiers from a list of timeseries file paths.

    Assumes that each filename contains the participant ID before the substring "_rois_".
    For example, "NYU_0050952_rois_cc200.1D" -> "NYU_0050952".

    Args:
        file_paths (List[str]): List of file paths to timeseries files.

    Returns:
        local_ids (List[str]): List of extracted local IDs corresponding to each file.
    """
    local_ids = []
    for path in file_paths:
        filename = os.path.basename(path)
        participant_id = filename.split("_rois_")[0]
        local_ids.append(participant_id)
    return local_ids


def create_phenotypic_lookup(pheno: pd.DataFrame) -> Dict[Tuple[str, str], int]:
    """
    Create a lookup dictionary mapping (site, participant) pairs to row indices in a 
    phenotypic DataFrame.

    Args:
        pheno (pd.DataFrame): Phenotypic DataFrame containing at least the columns:
            - 'SITE_ID': Site identifier for each participant.
            - 'participant': participant ID (numeric or string).

    Returns:
        pheno_dict (Dict[Tuple[str, str], int]): Dictionary where keys are (site, participant) 
        tuples and values are the corresponding row indices in the DataFrame. Both exact and 
        normalised site identifiers are included as keys.
    """
    pheno_dict = {}

    for i in range(len(pheno)):
        site = str(pheno.loc[i, 'SITE_ID']).upper()
        participant = str(int(pheno.loc[i, 'participant']))

        # Store with exact site name
        key = (site, participant)
        pheno_dict[key] = i

        # Also store with base site name for subsection matching
        base_site = normalise_site(site)
        if base_site != site:
            key_base = (base_site, participant)
            if key_base not in pheno_dict:
                pheno_dict[key_base] = i

    return pheno_dict


def match_participants(
    local_ids: List[str],
    pheno: pd.DataFrame,
    X: np.ndarray,
) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
    """
    Match local participant IDs to phenotypic data and filter the feature matrix accordingly.

    The output order matches the original phenotypic order.

    For each local ID, attempts to find the corresponding row in the phenotypic
    using both exact and normalised site identifiers. Converts diagnosis labels to binary
    (1=Autism, 0=Control) and extracts relevant metadata.

    Args:
        local_ids (List[str]): List of local participant identifiers.
        pheno (pd.DataFrame): Phenotypic DataFrame containing columns.
        X (np.ndarray): Feature matrix (samples x features) corresponding to local_ids.

    Returns:
        tuple: A tuple containing:
            - X_filtered (np.ndarray): Feature matrix filtered to matched participants, reordered 
            to match phenotypic order.
            - metadata (pd.DataFrame): Metadata DataFrame for matched participants with columns, 
            ordered to match phenotypic.
            - matched_indices (List[int]): List of indices in X that were successfully matched,
            in the order they appear in the output.
    """
    # Create reverse lookup: from local_id to its index in X
    local_id_to_x_index = {}
    for i, local_id in enumerate(local_ids):
        local_id_to_x_index[local_id] = i

    # Create lookups: (normalised_site, participant) -> local_id
    # Allows finding local_ids by their normalised site names
    local_id_lookup = {}
    for local_id in local_ids:
        site, participant_raw = parse_local_id(local_id)
        participant_normalised = normalise_participant_id(participant_raw)

        # Normalise the site name from the file (e.g., CMU_A -> CMU, MAXMUN_A -> MAX_MUN)
        normalised_site = normalise_site(site)

        # Store with normalised site name
        key = (normalised_site, participant_normalised)
        # Handle potential duplicates
        if key not in local_id_lookup:
            local_id_lookup[key] = local_id

    rows = []
    matched_indices = []
    unmatched_pheno = []

    # Iterate through phenotypic data in its original order
    for j in range(len(pheno)):
        site = str(pheno.loc[j, 'SITE_ID']).upper()
        participant = str(int(pheno.loc[j, 'subject']))

        found_local_id = None

        # Look up using the phenotypic site name directly
        key = (site, participant)
        if key in local_id_lookup:
            found_local_id = local_id_lookup[key]

        if found_local_id:
            # Get the index in X for this local_id
            x_index = local_id_to_x_index[found_local_id]

            # Convert DX_GROUP (1=Autism stays 1, 2=Control becomes 0)
            dx_raw = int(pheno.loc[j, 'DX_GROUP'])
            dx_converted = 1 if dx_raw == 1 else 0

            display_site = remove_numeric_suffix(site)

            rows.append({
                'ID': found_local_id,
                'SITE_ID': display_site,
                'DX_GROUP': dx_converted,
                'AGE_AT_SCAN': float(pheno.loc[j, 'AGE_AT_SCAN']),
                'SEX': int(pheno.loc[j, 'SEX']),
                'FIQ': float(pheno.loc[j, 'FIQ']) if pd.notna(pheno.loc[j, 'FIQ']) else np.nan,
                'VIQ': float(pheno.loc[j, 'VIQ']) if pd.notna(pheno.loc[j, 'VIQ']) else np.nan,
                'PIQ': float(pheno.loc[j, 'PIQ']) if pd.notna(pheno.loc[j, 'PIQ']) else np.nan,
                'HANDEDNESS_CATEGORY': pheno.loc[j, 'HANDEDNESS_CATEGORY']
                if pd.notna(pheno.loc[j, 'HANDEDNESS_CATEGORY']) else np.nan,
                'EYE_STATUS_AT_SCAN': pheno.loc[j, 'EYE_STATUS_AT_SCAN']
                if pd.notna(pheno.loc[j, 'EYE_STATUS_AT_SCAN']) else np.nan,
                'func_mean_fd': float(pheno.loc[j, 'func_mean_fd'])
                if pd.notna(pheno.loc[j, 'func_mean_fd']) else np.nan,
                'func_perc_fd': float(pheno.loc[j, 'func_perc_fd'])
                if pd.notna(pheno.loc[j, 'func_perc_fd']) else np.nan,
            })
            matched_indices.append(x_index)
        else:
            unmatched_pheno.append({
                'site': site,
                'participant': participant,
                'pheno_index': j
            })

    metadata = pd.DataFrame(rows)

    # Add the X index column (0-based)
    metadata.insert(0, 'X', range(len(metadata)))

    # Filter and reorder X to match the phenotypic order
    X_filtered = X[matched_indices]

    # Print summary
    print(f"Matched {len(metadata)} participants with phenotypic data")
    percentage = 100 * len(unmatched_pheno) / len(pheno)
    print(f"Unmatched phenotypic entries: {len(unmatched_pheno)} participants ({percentage:.1f}%)")

    return X_filtered, metadata, matched_indices

def filter_phenotypic_data(
    pheno: pd.DataFrame,
    sex: int | None = None,
    age_min: float | None = None,
    age_max: float | None = None,
) -> pd.DataFrame:
    """
    Filter phenotypic data by subgroup before processing.

    Filtering happens on the phenotypic DataFrame so that only matching
    participants are retained during the match_participants step. Files
    for excluded participants are loaded and preprocessed but dropped
    at the matching stage, so no re-downloading is required.

    Args:
        pheno (pd.DataFrame): Phenotypic DataFrame loaded from CSV.
        sex (int): Biological sex filter. 1 for Male, 2 for Female. If None, includes both.
        age_min (float): Minimum age (inclusive). If None, no lower bound.
        age_max (float): Maximum age (inclusive). If None, no upper bound.

    Returns:
        filtered (pd.DataFrame): Filtered phenotypic DataFrame.
    """
    mask = pd.Series(True, index=pheno.index)

    if sex is not None:
        mask &= pheno["SEX"] == sex

    if age_min is not None:
        mask &= pheno["AGE_AT_SCAN"] >= age_min

    if age_max is not None:
        mask &= pheno["AGE_AT_SCAN"] <= age_max

    filtered = pheno[mask].reset_index(drop=True)

    n_before = len(pheno)
    n_after = len(filtered)
    print(f"Subgroup filter: {n_before} → {n_after} participants retained")

    return filtered


def match_males_to_females(
    pheno: pd.DataFrame,
    female_ids: set,
    age_tolerance: float = 2.0,
    seed: int = 42,
    available_ids: set | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match male participants to females by site and age.

    For each female participant, finds the closest-age male from the same
    site within age_tolerance years. Each male can only be matched once.
    Females with no available male match at their site are dropped.

    Args:
        pheno (pd.DataFrame): Full phenotypic DataFrame (both sexes).
        female_ids (set): Subject IDs of the females to match against. Should be
            taken from female_metadata.csv so only females that survived
            preprocessing are included.
        age_tolerance (float): Maximum allowed age difference in years.
        seed (int): Random seed for breaking ties between equally close matches.
        available_ids (set): Optional set of integer subject IDs with valid
            preprocessed files. Males whose files were skipped during
            preprocessing are excluded from the candidate pool.

    Returns:
        tuple: a tuple containing:
            - matched_males (pd.DataFrame): Subset of the pheno DataFrame containing the 
            successfully matched male participants, reindexed and filtered by available IDs.
            - females (pd.DataFrame): The original subset of females from the pheno DataFrame 
            corresponding to the input female_ids, used as the reference for matching.
    """
    females = pheno[pheno["subject"].isin(female_ids)].copy()
    males = pheno[pheno["SEX"] == 1].copy()

    if available_ids is not None:
        n_before = len(males)
        males = males[males["subject"].isin(available_ids)]
        n_excluded = n_before - len(males)
        if n_excluded > 0:
            print(f"  Excluded {n_excluded} male(s) with missing/skipped files")

    available_male_ids = set(males["subject"].tolist())
    matched_male_ids = []
    unmatched_females = []

    # Shuffle females to avoid ordering bias when multiple matches tie
    females_shuffled = females.sample(frac=1, random_state=seed).reset_index(drop=True)

    for _, female in females_shuffled.iterrows():
        site_males = males[
            (males["SITE_ID"] == female["SITE_ID"]) &
            (males["subject"].isin(available_male_ids))
        ]

        if site_males.empty:
            unmatched_females.append(female["subject"])
            continue

        age_diffs = (site_males["AGE_AT_SCAN"] - female["AGE_AT_SCAN"]).abs()
        closest_idx = age_diffs.idxmin()

        if age_diffs[closest_idx] > age_tolerance:
            unmatched_females.append(female["subject"])
            continue

        matched_id = site_males.loc[closest_idx, "subject"]
        matched_male_ids.append(matched_id)
        available_male_ids.discard(matched_id)

    matched_males = males[males["subject"].isin(matched_male_ids)].reset_index(drop=True)

    print(f"  Female participants:       {len(females)}")
    print(f"  Matched males found:       {len(matched_males)}")
    print(f"  Females with no match:     {len(unmatched_females)}")
    if unmatched_females:
        print(f"  (age tolerance: ±{age_tolerance} years, same site required)")

    return matched_males, females
