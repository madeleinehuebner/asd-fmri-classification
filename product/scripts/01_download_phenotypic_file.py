"""
01_download_phenotypic_file.py

Downloads the ABIDE phenotypic CSV file (subject metadata) from the public FCP-INDI S3 bucket.
This file contains diagnostic labels (ASD vs Control), demographics, and site info for all subjects.

Code structure based on examples from:
Amazon Web Services (AWS) Boto3 Documentation S3.Client.download_file
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

Files are saved to /data/external/.

Usage:
    python 01_download_phenotypic_file.py
"""

import os
import argparse
import boto3
from botocore import UNSIGNED
from botocore.client import Config

from utils.paths import display_path

# Download phenotypic file
def download_phenotypic_file():
    """
    Download ABIDE phenotypic CSV from the FCP-INDI S3 bucket.
    """
    # S3 bucket and client
    bucket = "fcp-indi"
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    # S3 key and local file name
    key = "data/Projects/ABIDE_Initiative/Phenotypic_V1_0b_preprocessed.csv"
    local_name = "Phenotypic_V1_0b_preprocessed.csv"

    # Output path
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(root_dir, "data", "external")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, local_name)

    print(f"Downloading {local_name} from s3://{bucket}/{key} ...")

    try:
        s3.download_file(bucket, key, output_path)
        print(f"Downloaded successfully to {display_path(output_path)}")
    except Exception as e:
        print(f"Download failed: {e}")


if __name__ == "__main__":

    download_phenotypic_file()