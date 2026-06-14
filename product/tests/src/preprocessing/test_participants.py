"""Tests for preprocessing/participants.py."""

import numpy as np
import pandas as pd
import pytest

from preprocessing.participants import (
    create_phenotypic_lookup,
    extract_local_ids,
    filter_phenotypic_data,
    match_males_to_females,
    match_participants,
    normalise_participant_id,
    normalise_site,
    parse_local_id,
    remove_numeric_suffix,
)


@pytest.fixture
def pheno_df():
    """
    6 participants across 3 sites.
    Males  : 50952 NYU age 12 | 51100 NYU age 20 | 51300 YALE age 16
    Females: 51234 NYU age 13 | 51400 YALE age 17 | 51500 CALTECH age 22
    Subject 51500 has func_mean_fd=NaN.
    """
    return pd.DataFrame({
        "subject":              [50952,  51234,  51100,  51300,  51400,  51500],
        "participant":          [50952,  51234,  51100,  51300,  51400,  51500],
        "SITE_ID":              ["NYU",  "NYU",  "NYU",  "YALE", "YALE", "CALTECH"],
        "DX_GROUP":             [1,      2,      1,      1,      2,      2],
        "AGE_AT_SCAN":          [12.0,   13.0,   20.0,   16.0,   17.0,   22.0],
        "SEX":                  [1,      2,      1,      1,      2,      2],
        "FIQ":                  [110.0,  105.0,  115.0,  108.0,  np.nan, 112.0],
        "VIQ":                  [108.0,  103.0,  113.0,  106.0,  np.nan, 110.0],
        "PIQ":                  [112.0,  107.0,  117.0,  110.0,  np.nan, 114.0],
        "HANDEDNESS_CATEGORY":  ["R",    "R",    "L",    "R",    "R",    np.nan],
        "EYE_STATUS_AT_SCAN":   [1,      1,      1,      1,      np.nan, 1],
        "func_mean_fd":         [0.10,   0.15,   0.12,   0.20,   0.18,   np.nan],
        "func_perc_fd":         [5.0,    6.0,    4.5,    7.0,    6.5,    np.nan],
    })


@pytest.fixture
def local_ids():
    """Scrambled relative to pheno_df order."""
    return [
        "CALTECH_0051500", "YALE_0051400", "NYU_0050952",
        "YALE_0051300",    "NYU_0051234",  "NYU_0051100",
    ]


@pytest.fixture
def feature_matrix():
    np.random.seed(0)
    return np.random.randn(6, 10)


class TestNormaliseSite:

    @pytest.mark.parametrize("site,expected", [
        ("NYU", "NYU"), ("YALE", "YALE"), ("CALTECH", "CALTECH"),
        ("CMU_A", "CMU"), ("CMU_B", "CMU"), ("CMU", "CMU"),
        ("MAXMUN_A", "MAX_MUN"), ("MAXMUN_B", "MAX_MUN"), ("MAX_MUN", "MAX_MUN"),
    ])
    def test_site_normalisation(self, site, expected):
        assert normalise_site(site) == expected

    @pytest.mark.parametrize("site", ["LEUVEN_1", "LEUVEN_2", "UCLA_1", "UCLA_2", "UM_1", "UM_2"])
    def test_numeric_suffixes_are_preserved(self, site):
        assert normalise_site(site) == site

    def test_unknown_site_unchanged_or_letter_suffix_stripped(self):
        assert normalise_site("UNKNOWN_SITE") == "UNKNOWN_SITE"
        assert normalise_site("NEWSITE_X") == "NEWSITE"


class TestRemoveNumericSuffix:

    @pytest.mark.parametrize("site,expected", [
        ("LEUVEN_1", "LEUVEN"), ("UCLA_2", "UCLA"), ("UM_1", "UM"),
        ("CMU", "CMU"), ("NYU", "NYU"),
    ])
    def test_removes_or_preserves_suffix(self, site, expected):
        assert remove_numeric_suffix(site) == expected

    def test_letter_suffix_not_removed(self):
        assert remove_numeric_suffix("CMU_A") == "CMU_A"


class TestParseLocalId:

    @pytest.mark.parametrize("local_id,expected_site,expected_part", [
        ("NYU_0050952",      "NYU",      "0050952"),
        ("LEUVEN_1_0051234", "LEUVEN_1", "0051234"),
        ("MAX_MUN_0051234",  "MAX_MUN",  "0051234"),
    ])
    def test_parse_variants(self, local_id, expected_site, expected_part):
        site, participant = parse_local_id(local_id)
        assert site == expected_site
        assert participant == expected_part

    def test_site_is_uppercased(self):
        site, _ = parse_local_id("nyu_0050952")
        assert site == "NYU"


class TestNormaliseParticipantId:

    @pytest.mark.parametrize("raw,expected", [
        ("0050952", "50952"), ("50952", "50952"), ("000", "0"), ("007", "7"),
    ])
    def test_leading_zeros_removed(self, raw, expected):
        assert normalise_participant_id(raw) == expected


class TestExtractLocalIds:

    def test_extracts_ids_and_preserves_order(self, participants_file_paths, participants_local_ids):
        result = extract_local_ids(participants_file_paths)
        assert result == participants_local_ids

    def test_strips_directory_and_extension(self):
        paths = ["/some/deep/dir/NYU_0050952_rois_cc200.1D"]
        assert extract_local_ids(paths) == ["NYU_0050952"]

    def test_empty_input(self):
        assert extract_local_ids([]) == []


class TestCreatePhenotypicLookup:

    def test_all_participants_have_entries(self, pheno_df):
        lookup = create_phenotypic_lookup(pheno_df)
        for _, row in pheno_df.iterrows():
            site = str(row["SITE_ID"]).upper()
            participant = str(int(row["participant"]))
            assert (site, participant) in lookup

    def test_key_format_and_value_range(self, pheno_df):
        lookup = create_phenotypic_lookup(pheno_df)
        n_rows = len(pheno_df)
        for key, idx in lookup.items():
            assert isinstance(key, tuple) and len(key) == 2
            assert all(isinstance(part, str) for part in key)
            assert 0 <= idx < n_rows

    def test_normalised_site_key_also_stored(self):
        """Sites with subsections generate both raw and normalised keys."""
        pheno = pd.DataFrame({"SITE_ID": ["CMU_A"], "participant": [51001]})
        lookup = create_phenotypic_lookup(pheno)
        assert ("CMU_A", "51001") in lookup
        assert ("CMU", "51001") in lookup


class TestMatchParticipants:

    def test_output_structure_and_columns(self, pheno_df, local_ids, feature_matrix):
        X_filtered, metadata, matched_indices = match_participants(local_ids, pheno_df, feature_matrix)
        assert len(metadata) == len(pheno_df)
        expected_cols = {"X", "ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX",
                         "FIQ", "VIQ", "PIQ", "HANDEDNESS_CATEGORY", "EYE_STATUS_AT_SCAN",
                         "func_mean_fd", "func_perc_fd"}
        assert expected_cols.issubset(set(metadata.columns))
        assert list(metadata["X"]) == list(range(len(metadata)))

    def test_output_follows_pheno_order_and_dx_encoding(self, pheno_df, local_ids, feature_matrix):
        """Metadata is sorted by pheno order; DX_GROUP=2 is re-encoded to 0."""
        _, metadata, _ = match_participants(local_ids, pheno_df, feature_matrix)
        expected_ids = ["NYU_0050952", "NYU_0051234", "NYU_0051100",
                        "YALE_0051300", "YALE_0051400", "CALTECH_0051500"]
        assert list(metadata["ID"]) == expected_ids
        assert metadata.iloc[0]["DX_GROUP"] == 1   # ASD stays 1
        assert metadata.iloc[1]["DX_GROUP"] == 0   # Control → 0

    def test_x_reordered_to_pheno_order(self, pheno_df, local_ids, feature_matrix):
        """X_filtered is reordered: local_ids[2]=NYU_0050952 → pheno row 0."""
        X_filtered, _, _ = match_participants(local_ids, pheno_df, feature_matrix)
        np.testing.assert_array_equal(X_filtered[0], feature_matrix[2])
        np.testing.assert_array_equal(X_filtered[5], feature_matrix[0])

    def test_nan_values_preserved(self, pheno_df, local_ids, feature_matrix):
        """NaN in func_mean_fd and FIQ must survive in metadata."""
        _, metadata, _ = match_participants(local_ids, pheno_df, feature_matrix)
        assert pd.isna(metadata.iloc[5]["func_mean_fd"])   # 51500
        assert pd.isna(metadata.iloc[4]["FIQ"])            # 51400

    def test_unmatched_ids_excluded(self, pheno_df, feature_matrix):
        """Local IDs with no pheno record are silently excluded; unmatched pheno rows too."""
        local_ids_with_extra = ["NYU_0050952", "NYU_9999999"]
        X_small = feature_matrix[:2]
        pheno_one = pheno_df.iloc[[0]].reset_index(drop=True)
        _, metadata, _ = match_participants(local_ids_with_extra, pheno_one, X_small)
        assert "NYU_9999999" not in list(metadata["ID"])
        assert len(metadata) == 1


class TestFilterPhenotypicData:

    def test_no_filters_returns_all_rows(self, pheno_df):
        assert len(filter_phenotypic_data(pheno_df)) == len(pheno_df)

    def test_sex_filter(self, pheno_df):
        males   = filter_phenotypic_data(pheno_df, sex=1)
        females = filter_phenotypic_data(pheno_df, sex=2)
        assert len(males) == 3 and (males["SEX"] == 1).all()
        assert len(females) == 3 and (females["SEX"] == 2).all()

    def test_age_filters_inclusive(self, pheno_df):
        """age_min and age_max are inclusive bounds."""
        assert len(filter_phenotypic_data(pheno_df, age_min=15.0)) == 4
        assert len(filter_phenotypic_data(pheno_df, age_max=15.0)) == 2
        assert len(filter_phenotypic_data(pheno_df, age_min=12.0, age_max=12.0)) == 1

    def test_combined_filters_and_reset_index(self, pheno_df):
        result = filter_phenotypic_data(pheno_df, sex=1, age_min=15.0)
        assert len(result) == 2
        assert (result["SEX"] == 1).all() and (result["AGE_AT_SCAN"] >= 15.0).all()
        assert list(result.index) == list(range(len(result)))

    def test_no_match_returns_empty_dataframe(self, pheno_df):
        result = filter_phenotypic_data(pheno_df, age_min=100.0)
        assert len(result) == 0 and isinstance(result, pd.DataFrame)


class TestMatchMalesToFemales:
    """
    pheno_df layout:
      Males  : 50952 NYU age 12 | 51100 NYU age 20 | 51300 YALE age 16
      Females: 51234 NYU age 13 | 51400 YALE age 17 | 51500 CALTECH age 22

    With default age_tolerance=2.0 and female_ids={51234, 51400, 51500}:
      51234 (NYU, 13)    → matched to 50952 (NYU, 12)   diff=1 ✓
      51400 (YALE, 17)   → matched to 51300 (YALE, 16)  diff=1 ✓
      51500 (CALTECH, 22)→ no CALTECH males              → unmatched
    """

    def test_matched_males_and_females_structure(self, pheno_df):
        matched_males, females = match_males_to_females(pheno_df, {51234, 51400, 51500})
        assert isinstance(matched_males, pd.DataFrame) and isinstance(females, pd.DataFrame)
        assert len(matched_males) == 2
        assert (matched_males["SEX"] == 1).all()
        assert set(females["subject"]) == {51234, 51400, 51500}

    def test_tight_age_tolerance_excludes_all(self, pheno_df):
        matched_males, _ = match_males_to_females(pheno_df, {51234, 51400, 51500}, age_tolerance=0.5)
        assert len(matched_males) == 0

    def test_available_ids_restricts_candidates(self, pheno_df):
        """available_ids={51300} limits males to YALE only → only 51400 matched."""
        matched_males, _ = match_males_to_females(pheno_df, {51234, 51400, 51500}, available_ids={51300})
        assert len(matched_males) == 1
        assert 51300 in matched_males["subject"].values

    def test_each_male_matched_at_most_once_and_reproducible(self, pheno_df):
        m1, _ = match_males_to_females(pheno_df, {51234, 51400, 51500}, seed=0)
        m2, _ = match_males_to_females(pheno_df, {51234, 51400, 51500}, seed=0)
        assert m1["subject"].nunique() == len(m1)
        pd.testing.assert_frame_equal(m1.reset_index(drop=True), m2.reset_index(drop=True))
