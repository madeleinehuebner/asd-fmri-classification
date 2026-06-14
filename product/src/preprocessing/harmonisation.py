import numpy as np
import pandas as pd
from neuroCombat import neuroCombat

def prepare_combat_covariates(metadata):
    """
    Prepare covariates for ComBat harmonisation from phenotypic metadata.

    Extracts site, diagnosis, age, sex, and motion information from the metadata
    DataFrame and constructs a covariates DataFrame suitable for batch effect
    correction. Missing motion values are replaced with the mean.

    Args:
        metadata (pd.DataFrame): Phenotypic metadata containing columns:
            - 'SITE_ID': Site identifier for each subject (batch).
            - 'DX_GROUP': Diagnosis label (e.g., 0=Control, 1=Autism).
            - 'AGE_AT_SCAN': Age at scan in years.
            - 'SEX': Sex coded numerically (e.g., 0/1).
            - 'func_mean_fd': Mean framewise displacement (motion).

    Returns:
        pd.DataFrame: Covariates DataFrame with columns:
            - 'batch': Site identifiers.
            - 'diagnosis': Diagnosis labels.
            - 'age': Age at scan.
            - 'sex': Sex.
            - 'motion': Mean framewise displacement (NaNs replaced with mean).
    """
    sites = metadata['SITE_ID'].values
    diagnosis = metadata['DX_GROUP'].values
    age = metadata['AGE_AT_SCAN'].values
    sex = metadata['SEX'].values
    motion = metadata['func_mean_fd'].fillna(metadata['func_mean_fd'].mean()).values

    # Create covariates dataframe
    covars = pd.DataFrame({
        'batch': sites,
        'diagnosis': diagnosis,
        'age': age,
        'sex': sex,
        'motion': motion
    })

    print("\nCovariates prepared:")
    print(f"  Sites (batch): {len(np.unique(sites))} unique values")
    print(f"  Diagnosis: {len(np.unique(diagnosis))} classes")
    print(f"  Age range: [{age.min():.1f}, {age.max():.1f}]")
    print(f"  Sex: {len(np.unique(sex))} categories")
    print(f"  Motion: [{motion.min():.3f}, {motion.max():.3f}]")

    return covars

def apply_harmonisation(X, metadata):
    """
    Apply ComBat harmonisation to remove site-related batch effects from the feature matrix.

    Uses phenotypic covariates (site, diagnosis, age, sex, motion) to adjust the data
    while preserving biological or clinical variability of interest.

    Args:
        X (np.ndarray): Feature matrix of shape (n_samples, n_features) to be harmonised.
        metadata (pd.DataFrame): Phenotypic metadata corresponding to each sample.

    Returns:
        X_harmonised (np.ndarray): Harmonised feature matrix of the same shape as X (n_samples, 
        n_features).
    """
    # Apply ComBat
    print("\nRunning ComBat harmonisation...")
    print("This may take several minutes...")

    covars = prepare_combat_covariates(metadata)

    combat_result = neuroCombat(
        dat=X.T,
        covars=covars,
        batch_col='batch',
        categorical_cols=['diagnosis', 'sex'],
        continuous_cols=['age', 'motion']
    )

    X_harmonised = combat_result['data'].T

    return X_harmonised
