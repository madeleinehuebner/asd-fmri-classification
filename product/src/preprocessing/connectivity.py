"""
Functional Connectivity Feature Extraction Module

This module provides utilities for computing functional connectivity matrices 
from preprocessed fMRI time-series data using the CC200 (Craddock 200) atlas. 
It handles the transformation of correlation coefficients, ROI label 
mapping, and the generation of descriptive feature names for pairwise 
interactions.

Main Workflow:
    1. Load anatomical ROI labels from a provided CSV.
    2. Compute correlation matrices for a cohort of participants.
    3. Vectorise the upper triangle of the matrices.
    4. Apply Fisher z-transformation to normalise the correlation distribution.
    5. Map ROI indices back to anatomical names (e.g., 'Left Precuneus <-> Thalamus').
"""


import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from nilearn.connectome import ConnectivityMeasure
from nilearn.plotting import find_parcellation_cut_coords
import pandas as pd

def load_roi_labels(cc200_map: str, aal_map: str) -> pd.DataFrame:
    """
    Loads CC200 labels and merges them with a standard AAL anatomical mapping.

    This function extracts the primary AAL region from the complex CC200 string 
    format. If the primary region is 'None', it searches for the next available 
    anatomical label in the string to ensure valid mapping where possible.

    Args:
        cc200_map: Path to the CC200_ROI_labels.csv file.
        aal_map: Path to the AAL regions mapping CSV containing 
            'ANATOMICAL DESCRIPTION', 'LABEL', and 'ABBREVIATION'.

    Returns:
        A pandas DataFrame where each CC200 ROI index is enriched with 
        AAL labels and abbreviations.
    """
    cc200_df = pd.read_csv(cc200_map)
    aal_map_df = pd.read_csv(aal_map)
    
    def extract_aal_key(label_string):
        if pd.isna(label_string):
            return "None"
        
        # Regex to find all regions: ["Region_Name": 0.XX]
        regions = re.findall(r'\["([^"]+)":', str(label_string))
        
        # Select the first region that isn't "None"
        for region in regions:
            if region != "None":
                return region
        
        return "None"

    # Create the key used to join with AAL map
    cc200_df['aal_key'] = cc200_df['AAL'].apply(extract_aal_key)
    
    # Merge CC200 with the AAL map
    merged_df = cc200_df.merge(
        aal_map_df, 
        left_on='aal_key', 
        right_on='LABEL', 
        how='left'
    )
    
    fill_values = {
        'ANATOMICAL DESCRIPTION': 'Unknown Region',
        'NOTATION': 'Unknown Region',
        'ABBREVIATION': 'U_R',
        'NETWORK': 'Unknown Network'
    }
    
    return merged_df.fillna(value=fill_values)

def generate_feature_labels(
    n_rois: int,
    cc200_map: str,
    aal_map: str
) -> pd.DataFrame:
    """
    Generates descriptive labels for all pairwise connectivity features.

    Creates a mapping for the upper triangle of a connectivity matrix. For 
    each pair of ROIs (i, j), it generates multiple name formats including 
    raw AAL strings, full anatomical descriptions, and short abbreviations.

    Args:
        n_rois: The number of ROIs in the parcellation (e.g., 200).
        cc200_map: Path to the CC200_ROI_labels.csv file.
        aal_map: Path to the AAL regions mapping CSV.

    Returns:
        A DataFrame containing columns for indices, raw labels, anatomical 
        descriptions, and abbreviations for every pairwise connection.
    """
    labels_df = load_roi_labels(cc200_map, aal_map)
    roi_labels = []
    feature_idx = 0

    for i in range(n_rois):
        for j in range(i + 1, n_rois):
            row_i = labels_df.iloc[i]
            row_j = labels_df.iloc[j]

            def connect(val_i, val_j):
                return f"{val_i} <-> {val_j}"
            
            roi_labels.append({
                'feature_idx': feature_idx,
                'roi_i': i,
                'roi_j': j,
                'desc_i': row_i['ANATOMICAL DESCRIPTION'],
                'desc_j': row_j['ANATOMICAL DESCRIPTION'],
                'raw_i': row_i['AAL'],
                'raw_j': row_j['AAL'],
                'label_i': row_i['NOTATION'],
                'label_j': row_j['NOTATION'],
                'abbrev_i': row_i['ABBREVIATION'],
                'abbrev_j': row_j['ABBREVIATION'],
                'network_i': row_i['NETWORK'],
                'network_j': row_j['NETWORK'],
                'desc_feature_name': connect(row_i['ANATOMICAL DESCRIPTION'], row_j['ANATOMICAL DESCRIPTION']),
                'feature_name': connect(row_i['AAL'], row_j['AAL']),
                'label_feature_name': connect(row_i['NOTATION'], row_j['NOTATION']),
                'abbrev_feature_name': connect(row_i['ABBREVIATION'], row_j['ABBREVIATION']),
                'network': connect(row_i['NETWORK'], row_j['NETWORK']),
            })
            feature_idx += 1

    return pd.DataFrame(roi_labels)

def compute_connectivity_matrices(
    cleaned_time_series: List[np.ndarray]
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Computes functional connectivity matrices from fMRI time-series.

    Calculates the Pearson correlation coefficient between all pairs of ROIs, 
    vectorises the upper triangle (excluding the diagonal), and applies 
    a Fisher z-transformation to normalise the distribution of coefficients.

    Args:
        cleaned_time_series: A list of 2D numpy arrays where each array 
            represents a participant's time-series (shape: [timepoints, ROIs]).

    Returns:
        A 2D numpy array (shape: [n_participants, n_features]) containing 
        the Fisher z-transformed connectivity features.
    """
    correlation_measure = ConnectivityMeasure(
        kind='correlation',
        standardize='zscore_sample',
        vectorize=True,
        discard_diagonal=True
    )

    X = correlation_measure.fit_transform(cleaned_time_series)

    # Fisher z-transformation with clipping
    X = np.clip(X, -0.999999, 0.999999)
    X = np.arctanh(X)

    print(f"Connectivity matrix shape: {X.shape}")
    
    return X


def generate_coords_labels(
    cc200_map: str, 
    aal_map: str, 
    atlas: str
) -> pd.DataFrame:

    coords_array = find_parcellation_cut_coords(atlas)

    labels_df = load_roi_labels(cc200_map, aal_map)
    
    # Extract coordinates from the atlas
    coords_array = find_parcellation_cut_coords(atlas)
    coords_df = pd.DataFrame(coords_array, columns=['x', 'y', 'z'])
    
    coords_df['label'] = labels_df['NOTATION']

    return coords_df