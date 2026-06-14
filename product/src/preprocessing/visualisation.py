import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Optional
from scipy.spatial import ConvexHull

def print_data_summary(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame
) -> None:
    """
    Print a summary of the dataset, including feature matrix shape, label distribution,
    and site metadata statistics.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features).
        y (np.ndarray): Array of target labels of shape (n_samples,). 0 = Control, 1 = Autism.
        metadata (pd.DataFrame): DataFrame containing metadata for each sample.
    """
    print("\n" + "="*60)
    print("Dataset Summary")
    print("="*60)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    
    print(f"\n--- Site Distribution ---")
    site_counts = metadata['SITE_ID'].value_counts().sort_index()
    print(site_counts)
    
    print(f"\n--- Diagnosis Distribution ---")
    n_autism = np.sum(y == 1)
    n_control = np.sum(y == 0)
    total = len(y)
    print(f"Autism (DX_GROUP=1):  {n_autism} ({100*n_autism/total:.1f}%)")
    print(f"Control (DX_GROUP=0): {n_control} ({100*n_control/total:.1f}%)")
    
    if n_control > 0 and n_autism > 0:
        print("\nBoth classes present. Ready for classification.")
    else:
        print("\nError: Missing one class.")