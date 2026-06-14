from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_download_side_effect(downloaded: list):
    def _side_effect(bucket, key, path):
        downloaded.append(Path(path).name)
    return _side_effect


def _run_main(module, s3_keys, *, site=None, n=None, seed=42,
              age_min=None, age_max=None, sex=None):
    """Run main() with mocked S3 and return downloaded filenames."""
    mock_s3 = MagicMock()
    downloaded = []
    mock_s3.download_file.side_effect = _make_download_side_effect(downloaded)
    with (
        patch("boto3.client", return_value=mock_s3),
        patch.object(module, "list_all_keys", return_value=s3_keys),
        patch.object(module, "preprocess_timeseries", return_value=MagicMock()),
        patch.object(module, "display_path", return_value="<test-path>"),
        patch.object(module, "np"),
    ):
        module.main(site=site, n=n, seed=seed,
                    age_min=age_min, age_max=age_max, sex=sex)
    return downloaded


class TestListAllKeys:

    def test_single_page(self, module):
        """Returns all keys when the response is not truncated."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [{"Key": "a/b/c.1D"}, {"Key": "a/b/d.1D"}],
            "IsTruncated": False,
        }
        assert module.list_all_keys(mock_s3, "bucket", "a/b/") == ["a/b/c.1D", "a/b/d.1D"]

    def test_paginated(self, module):
        """Follows continuation tokens until IsTruncated is False."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "a/1.1D"}], "IsTruncated": True, "NextContinuationToken": "tok1"},
            {"Contents": [{"Key": "a/2.1D"}], "IsTruncated": False},
        ]
        result = module.list_all_keys(mock_s3, "bucket", "a/")
        assert result == ["a/1.1D", "a/2.1D"]
        assert mock_s3.list_objects_v2.call_count == 2

    def test_empty_bucket(self, module):
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {"IsTruncated": False}
        assert module.list_all_keys(mock_s3, "bucket", "empty/") == []


class TestMainErrors:

    def test_missing_pheno_csv_raises(self, module, tmp_path):
        with pytest.raises(SystemExit, match="Phenotypic CSV not found"):
            module.main(site=None, n=None, seed=None, age_min=None, age_max=None, sex=None)

    def test_no_matching_files_raises(self, module, pheno_csv):
        with (
            patch("boto3.client", return_value=MagicMock()),
            patch.object(module, "list_all_keys", return_value=[]),
        ):
            with pytest.raises(SystemExit, match="No matching ROI"):
                module.main(site=None, n=None, seed=None, age_min=None, age_max=None, sex=None)

    def test_site_filter_no_match_raises(self, module, pheno_csv, s3_keys_all):
        with (
            patch("boto3.client", return_value=MagicMock()),
            patch.object(module, "list_all_keys", return_value=s3_keys_all),
        ):
            with pytest.raises(SystemExit, match="No matching ROI"):
                module.main(site=["UNKNOWN"], n=None, seed=None, age_min=None, age_max=None, sex=None)


class TestMainFiltering:

    def test_no_filters_downloads_all(self, module, pheno_csv, s3_keys_all):
        assert len(_run_main(module, s3_keys_all)) == 6

    def test_site_filter_nyu(self, module, pheno_csv, s3_keys_all):
        downloaded = _run_main(module, s3_keys_all, site=["NYU"])
        assert len(downloaded) == 3
        assert all("NYU" in f for f in downloaded)

    def test_site_filter_multiple_and_case_insensitive(self, module, pheno_csv, s3_keys_all):
        """Multiple sites and lowercase site name both work."""
        assert len(_run_main(module, s3_keys_all, site=["NYU", "YALE"])) == 5
        assert len(_run_main(module, s3_keys_all, site=["nyu"])) == 3

    def test_sex_filters(self, module, pheno_csv, s3_keys_all):
        """sex=1 keeps 3 males; sex=2 keeps 3 females."""
        assert len(_run_main(module, s3_keys_all, sex=1)) == 3
        assert len(_run_main(module, s3_keys_all, sex=2)) == 3

    def test_age_filters(self, module, pheno_csv, s3_keys_all):
        """age_min, age_max, and combined range all exclude correctly."""
        assert len(_run_main(module, s3_keys_all, age_min=15.0)) == 4
        assert len(_run_main(module, s3_keys_all, age_max=18.0)) == 3
        assert len(_run_main(module, s3_keys_all, age_min=13.0, age_max=20.0)) == 3

    def test_combined_site_sex_age_filter(self, module, pheno_csv, s3_keys_all):
        """All three filters compose correctly; returns the one matching participant."""
        downloaded = _run_main(module, s3_keys_all, site=["NYU"], age_min=15.0, sex=2)
        assert len(downloaded) == 1
        assert "0051002" in downloaded[0]


class TestMainSampling:

    def test_n_limits_and_seed_reproducibility(self, module, pheno_csv, s3_keys_all):
        """n caps downloads; same seed gives identical results."""
        assert len(_run_main(module, s3_keys_all, n=2, seed=0)) == 2
        first  = _run_main(module, s3_keys_all, n=3, seed=99)
        second = _run_main(module, s3_keys_all, n=3, seed=99)
        assert first == second

    def test_n_larger_than_pool_downloads_all(self, module, pheno_csv, s3_keys_all):
        assert len(_run_main(module, s3_keys_all, n=100, seed=0)) == 6

    def test_different_seeds_differ(self, module, pheno_csv, s3_keys_all):
        results = {tuple(sorted(_run_main(module, s3_keys_all, n=3, seed=s))) for s in range(20)}
        assert len(results) > 1


class TestMainPhenoMapEdgeCases:

    def _run(self, module, s3_keys):
        mock_s3 = MagicMock()
        downloaded = []
        mock_s3.download_file.side_effect = _make_download_side_effect(downloaded)
        with (
            patch("boto3.client", return_value=mock_s3),
            patch.object(module, "list_all_keys", return_value=s3_keys),
            patch.object(module, "preprocess_timeseries", return_value=MagicMock()),
            patch.object(module, "display_path", return_value="<test-path>"),
            patch.object(module, "np"),
        ):
            module.main(site=None, n=None, seed=None, age_min=None, age_max=None, sex=None)
        return downloaded

    def test_subject_not_in_pheno_map_is_skipped(self, module, tmp_path, make_pheno_df, make_s3_key):
        ext_dir = tmp_path / "data" / "external"
        ext_dir.mkdir(parents=True, exist_ok=True)
        make_pheno_df([{"SUB_ID": "51001", "SITE_ID": "NYU", "AGE_AT_SCAN": 12.0, "SEX": 1}]).to_csv(
            ext_dir / module.PHENO_CSV, index=False
        )
        downloaded = self._run(module, [make_s3_key("NYU", "0051001"), make_s3_key("NYU", "0099999")])
        assert len(downloaded) == 1
        assert "0051001" in downloaded[0]

    def test_invalid_filename_pattern_is_skipped(self, module, tmp_path, make_pheno_df, make_s3_key):
        ext_dir = tmp_path / "data" / "external"
        ext_dir.mkdir(parents=True, exist_ok=True)
        make_pheno_df([{"SUB_ID": "51001", "SITE_ID": "NYU", "AGE_AT_SCAN": 12.0, "SEX": 1}]).to_csv(
            ext_dir / module.PHENO_CSV, index=False
        )
        keys = [
            make_s3_key("NYU", "0051001"),
            "data/Projects/ABIDE_Initiative/Outputs/cpac/filt_noglobal/rois_cc200/README.txt",
        ]
        assert len(self._run(module, keys)) == 1
