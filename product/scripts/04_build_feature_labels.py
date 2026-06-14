"""
Generate and save feature labels and ROI coordinates for the CC200 atlas.

Maps each of the 200 CC200 ROIs to its corresponding AAL region, producing
a pairwise feature label table (19,900 connectivity features) and a coordinate
table for spatial visualisation. Both are saved to the processed data directory.

Args:
    cc200_map    (str): Path to CC200_ROI_labels.csv. Maps ROI indices to CC200 region names.
    aal_map      (str): Path to AAL_map.csv. Maps CC200 ROIs to AAL anatomical labels.
    atlas        (str): Path to cc200_roi_atlas.nii.gz. NIfTI atlas image used to extract MNI coordinates.
    processed_dir(str): Output directory for saved feature labels and coordinates.

Outputs:
    feature_labels.csv   One row per connectivity feature (ROI pair), with CC200 and AAL labels for each endpoint.
    coords.csv           One row per ROI, with MNI coordinates and AAL anatomical label.
"""


from pathlib import Path
from nilearn import image
from data_io.save_load_dataset import save_dataset
from preprocessing.connectivity import generate_coords_labels, generate_feature_labels
from utils.paths import get_project_root

def main(
    cc200_map: str,
    aal_map: str,
    atlas: str,
    processed_dir: str,
):

    feature_df = generate_feature_labels(200, cc200_map, aal_map)
    print(f"Generated connectivity features: \n{feature_df}")
    
    coords_df = generate_coords_labels(cc200_map,aal_map,atlas)
    print(f"Generated ROI coordinates and labels: \n{coords_df}")
    
    # Save both
    print("\nSaving Feature Labels and Coordinates...")
    processed_dir = Path(processed_dir)
    save_dataset(processed_dir=processed_dir, feature_labels=feature_df, coords=coords_df)

if __name__ == "__main__":
    
    ROOT = get_project_root()
    DATA = ROOT / "product" / "data"
    PROC = DATA / "processed"
    EXT = DATA / "external"
    
    CC200_MAP = str(EXT / "CC200_ROI_labels.csv")
    AAL_MAP = str(EXT / "AAL_map.csv")
    ATLAS = str(EXT / "cc200_roi_atlas.nii.gz")
    
    main(CC200_MAP, AAL_MAP, ATLAS, PROC)