"""
02_download_participants.py

Data Source:
- Craddock et al., 2013, The Neuro Bureau Preprocessing Initiative
    (Frontiers in Neuroinformatics, Neuroinformatics 2013)
    DOI: 10.3389/conf.fninf.2013.09.00041

This script uses the AWS Boto3 S3 client for downloading public ABIDE
preprocessed data from the public S3 bucket.

Files are saved to /data/raw/.

Code structure based on examples from:
Amazon Web Services (AWS) Boto3 Documentation S3.Client.download_file
https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html

Assumes filenames like: <SITE>_<SUBJECTID>_rois_cc200.1D
Example: NYU_0051002_rois_cc200.1D

Usage:
    python 02_download_participants.py --n 20              # set number of participants to download
    python 02_download_participants.py --site NYU --n 20   # set site filter
    python 02_download_participants.py --site NYU --n 20 --seed 42  # set random seed
    python 02_download_participants.py --site NYU --n 20 --seed 42 --age-min 13 --age-max 19 # set 
    age range
    python 02_download_participants.py --sex 2 # female participants
    python 02_download_participants.py --sex 1 # male participants
"""


import argparse
import random
import re
import boto3
from botocore.client import Config
from botocore import UNSIGNED
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocessing.participants import normalise_site
from preprocessing.timeseries import preprocess_timeseries
from utils.paths import display_path, get_project_root

BUCKET = "fcp-indi"
PREFIX = "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/"
FNAME_RE = re.compile(r"^([A-Za-z0-9_]+?)_(\d+)_rois_cc200\.1D$")

ROOT = get_project_root() / "product"
EXT = ROOT / "data" / "external"

sys.path.insert(0, str(ROOT / "src"))

PHENO_CSV = "Phenotypic_V1_0b_preprocessed.csv"

VALID_SEX_VALUES = {1, 2}

def list_all_keys(
    s3: boto3.client,
    bucket: str,
    prefix: str,
):
    """
    Recursively lists all object keys in an S3 bucket under a specific prefix.

    Args:
        s3 (boto3.client): An initialised boto3 S3 client.
        bucket (str): The name of the S3 bucket.
        prefix (str): The folder path/prefix to filter by.

    Returns:
        A list of strings representing the full S3 keys found.
    """
    keys, kwargs = [], {"Bucket": bucket, "Prefix": prefix}
    while True:
        resp = s3.list_objects_v2(**kwargs)
        keys += [o["Key"] for o in resp.get("Contents", [])]
        if resp.get("IsTruncated"):
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        else:
            break
    return keys


def main(
    site: str | None,
    n: int | None,
    seed: int | None,
    age_min: float | None,
    age_max: float | None,
    sex: int | None,
    subdir: str | None = None,
):
    """
    Filters and downloads ABIDE ROI files from S3 based on site, age, and count.

    This function performs the following steps:
    1. Validates the existence of the phenotypic CSV for metadata filtering.
    2. Connects to the FCP-INDI S3 bucket using an unsigned config (public access).
    3. Matches S3 objects against the CC200 ROI filename pattern.
    4. Filters participants by site name, age range, and sex using the phenotypic map.
    5. Randomly samples N participants (if specified) and downloads files to /data/raw/.

    Args:
        site (str | None): List of site identifiers (e.g., ['NYU', 'YALE']) to include.
        n (int | None): Number of participants to download. If None, downloads all matching.
        seed (int | None): Random seed for reproducible sampling.
        age_min (float | None): Minimum age threshold (inclusive) for participants.
        age_max (float | None): Maximum age threshold (inclusive) for participants.
        sex (int | None): Filter by sex. 1 for Male, 2 for Female. If None, includes both.
        subdir (str | None): Optional subfolder name within data/raw/ to store files.

    Raises:
        SystemExit: If the phenotypic CSV is missing or no files match the criteria.
    """
    outdir = ROOT / "data" / "raw" / subdir if subdir else ROOT / "data" / "raw"
    outdir.mkdir(parents=True, exist_ok=True)

    # Load phenotypic data
    pheno_path = EXT / PHENO_CSV
    if not pheno_path.exists():
        raise SystemExit(f"Phenotypic CSV not found: {pheno_path}")
    pheno = pd.read_csv(pheno_path)

    # Create mapping: (site, subject_id) → age
    pheno["SUB_ID"] = pheno["SUB_ID"].astype(str).str.zfill(7)
    pheno_map = {
        (row["SITE_ID"].upper(), row["SUB_ID"]): {
            "age": row["AGE_AT_SCAN"],
            "sex": row["SEX"],
        }
        for _, row in pheno.iterrows()
    }

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    keys = list_all_keys(s3, BUCKET, PREFIX)

    # Filter .1D files by site, age, and sex
    entries = []
    for k in keys:
        fname = k.split("/")[-1]
        m = FNAME_RE.match(fname)
        if not m:
            continue
        site_id, sub_id = normalise_site(m.group(1).upper()), m.group(2).zfill(7)

        # Filter by site (if specified)
        if site is not None and site_id not in [s.upper() for s in site]:
            continue

        participant = pheno_map.get((site_id, sub_id))
        if participant is None:
            continue

        # Filter by age range (if specified)
        age = participant["age"]
        if (age_min is not None and age < age_min) or (age_max is not None and age > age_max):
            continue

        # Filter by sex (if specified)
        if sex is not None and participant["sex"] != sex:
            continue

        entries.append((k, fname))

    if not entries:
        raise SystemExit(
            "No matching ROI .1D files found. Check filters.")

    # Skip files already present in outdir
    existing = {f.name for f in outdir.glob("*.1D")}
    missing = [(k, f) for k, f in entries if f not in existing]
    print(
        f"Already downloaded: {len(existing)} files. {len(missing)} remaining")

    if not missing:
        print("Nothing to download.")
        return

    if seed is not None:
        random.seed(seed)
    random.shuffle(missing)

    if n is None:
        n = len(missing)

    valid, invalid = [], []

    with tqdm(total=n, unit="file", desc="Downloading") as pbar:
        for key, fname in missing:
            if len(valid) >= n:
                break
            dest = outdir / fname
            s3.download_file(BUCKET, key, str(dest))
            ts = np.genfromtxt(dest)
            if preprocess_timeseries(ts) is not None:
                valid.append(fname)
                pbar.update(1)
            else:
                dest.unlink()
                invalid.append(fname)

    print(f"Downloaded {len(valid)} valid files to {display_path(outdir)}")
    if invalid:
        print(f"Skipped {len(invalid)} invalid files:")
        for fname in invalid:
            print(f"  {fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Download random ABIDE *_rois_cc200.1D files from S3.")
    p.add_argument("--site", nargs="+", type=str, default=None,
                help="Optional site filter(s)(e.g., NYU TRINITY YALE)")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--age-min", type=float, default=None)
    p.add_argument("--age-max", type=float, default=None)
    p.add_argument("--sex", type=int, default=None, choices=[1, 2])
    p.add_argument("--subdir", type=str, default=None,
                help="Optional subfolder within data/raw/ to store downloaded files.")
    args = p.parse_args()
    main(args.site, args.n, args.seed, args.age_min, args.age_max, args.sex, args.subdir)
