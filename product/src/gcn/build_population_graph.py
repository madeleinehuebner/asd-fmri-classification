"""
Population graph construction.

This module implements the population graph construction logic as defined in:
1. Parisot et al. "Spectral Graph Convolutions for Population-Based Disease 
   Prediction" MICCAI 2017.
2. Parisot et al. "Disease Prediction using Graph Convolutional Networks: 
   Application to Autism Spectrum Disorder and Alzheimer's Disease" 
   Medical Image Analysis, 2018.

The graph defines participants as nodes and edges based on a combination of 
feature similarity and phenotypic overlap (e.g., Sex, Age, Site).
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import torch

# Columns treated as quantitative.
# All other columns are treated as categorical.
QUANTITATIVE_SCORES = {'AGE_AT_SCAN', 'FIQ'}

def compute_phenotypic_gamma(
    participant_list: np.ndarray,
    metadata: pd.DataFrame,
    scores: list[str],
    id_col: str = 'ID',
    age_threshold: float = 2.0,
) -> np.ndarray:
    """
    Builds the phenotypic contribution to the adjacency matrix.

    Calculates gamma_total(v, w) = sum_h gamma_h(Mh(v), Mh(w)).
    For categorical measures, gamma is 1 if values are equal.
    For quantitative measures, gamma is 1 if the absolute difference is 
    less than age_threshold.

    Args:
        participant_list: Array of participant ID strings of shape (N,).
        metadata: DataFrame containing phenotypic information.
        scores: List of phenotypic column names to include.
        id_col: Name of the participant ID column in metadata.
        age_threshold: The theta threshold for quantitative measures.

    Returns:
        A float array of shape (N, N) representing the phenotypic graph.
    """
    # Index metadata by participant ID for fast lookup
    meta_indexed = metadata.set_index(id_col)

    N = len(participant_list)
    graph = np.zeros((N, N))

    for score_name in scores:
        # Extract values in participant_list order
        values = meta_indexed.loc[participant_list, score_name].values

        if score_name in QUANTITATIVE_SCORES:
            # Vectorised unit-step: 1 if |Mh(v) - Mh(w)| < theta
            vals_float = values.astype(float)
            diff = np.abs(vals_float[:, None] - vals_float[None, :])
            contribution = (diff < age_threshold).astype(float)
        else:
            # Vectorised Kronecker delta: 1 if Mh(v) == Mh(w)
            _vals = np.array(values)
            contribution = (_vals[:, None] == _vals[None, :]).astype(float)

        # Zero out diagonal
        np.fill_diagonal(contribution, 0)
        graph += contribution

    return graph


def compute_similarity(features: np.ndarray, sigma: float = None) -> np.ndarray:
    """
    Calculates the similarity matrix between participant features.

    Implements Sim(v, w) = exp(-[rho(x(v), x(w))]^2 / (2 * sigma^2)),
    where rho is the correlation distance.

    Args:
        features: Feature matrix of shape (N, C).
        sigma: Kernel width. If None, defaults to the mean pairwise distance.

    Returns:
        A float array of shape (N, N) representing participant similarities.
    """
    dist = cdist(features, features, metric='correlation')
    if sigma is None:
        sigma = np.mean(dist)
    return np.exp(-(dist ** 2) / (2 * sigma ** 2))


def build_population_graph(
    features: np.ndarray,
    metadata: pd.DataFrame,
    phenotypic_scores: list[str],
    id_col: str = 'ID',
    age_threshold: float = 2.0,
    sigma: float = None,
) -> np.ndarray:
    """
    Builds the population graph adjacency matrix W.

    Combines feature similarity and phenotypic gamma:
    W(v, w) = Sim(v, w) * sum_h gamma_h(Mh(v), Mh(w)).

    Args:
        features: Feature matrix or tensor of shape (N, C). Row order must 
        match the metadata.
        metadata: DataFrame produced by load_dataset().
        phenotypic_scores: List of phenotypic column names.
        id_col: Participant ID column name in metadata.
        age_threshold: Threshold for quantitative phenotypic measures.
        sigma: Kernel width. If None, set to mean pairwise distance.

    Returns:
        A float adjacency matrix W of shape (N, N).
    """
    
    participant_list = metadata['ID'].astype(str).values
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    
    gamma = compute_phenotypic_gamma(
        participant_list=participant_list,
        metadata=metadata,
        scores=phenotypic_scores,
        id_col=id_col,
        age_threshold=age_threshold,
    )
    sim = compute_similarity(features, sigma=sigma)
    return sim * gamma

def threshold_graph(W: np.ndarray, percentile: float = 70.0) -> np.ndarray:
    """
    Zeros out edges below a given percentile of non-zero weights.

    This ensures graph sparsity and removes weak connections while 
    preserving symmetry.

    Args:
        W: Adjacency matrix from build_population_graph of shape (N, N).
        percentile: Edges below this percentile of non-zero values are removed.

    Returns:
        A thresholded adjacency matrix of shape (N, N).
    """
    non_zero = W[W > 0]
    tau = np.percentile(non_zero, percentile)
    W_thresh = np.where(W >= tau, W, 0.0)
    np.fill_diagonal(W_thresh, 0)  # ensure no self-loops
    return W_thresh