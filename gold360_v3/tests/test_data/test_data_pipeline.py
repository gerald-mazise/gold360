import numpy as np
import pandas as pd
import pytest

from gold360.data.loader import DataLoader
from gold360.data.validator import DataValidator
from gold360.data.temporal import TemporalAligner


class TestDataLoader:
    def test_initialization(self):
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, "load_all")

    def test_load_types(self):
        loader = DataLoader()
        result = loader.list_datasets()
        assert isinstance(result, list)

    def test_source_info(self):
        loader = DataLoader()
        info = loader.get_source_info("fgr_deliveries")
        if info:
            assert "path" in info
            assert "type" in info
            assert "frequency" in info


class TestDataValidator:
    def test_initialization(self):
        validator = DataValidator()
        assert validator is not None

    def test_validate_dataframe(self, sample_mine_data):
        validator = DataValidator()
        result = validator.validate(sample_mine_data, "test_data")
        assert isinstance(result, object)

    def test_missing_check(self, sample_mine_data):
        df = sample_mine_data.copy()
        df.loc[0, "delivery_volume_kg"] = np.nan
        validator = DataValidator()
        result = validator.validate(df, "test_missing")
        assert any(c["check"] == "missing_values" for c in result.checks)

    def test_duplicate_check(self, sample_mine_data):
        df = pd.concat([sample_mine_data, sample_mine_data.iloc[:5]], ignore_index=True)
        validator = DataValidator()
        result = validator.validate(df, "test_duplicates")
        assert any(c["status"] == "FAIL" for c in result.checks if "duplicate" in c["check"])

    def test_outlier_detection(self, sample_mine_data):
        df = sample_mine_data.copy()
        df.loc[0, "delivery_volume_kg"] = 1e6
        validator = DataValidator()
        result = validator.validate(df, "test_outliers")
        assert any("outlier" in c["check"] for c in result.checks)


class TestTemporalAligner:
    def test_initialization(self):
        aligner = TemporalAligner()
        assert aligner is not None

    def test_quarterly_to_monthly(self):
        aligner = TemporalAligner()
        df = pd.DataFrame({
            "period": pd.Series(["2020-Q1", "2020-Q2", "2020-Q3", "2020-Q4"], dtype=str),
            "value": [100, 110, 105, 115],
        })
        result = aligner.align_dataset(df, {"frequency": "quarterly"}, "test")
        assert len(result) == 12

    def test_annual_to_monthly(self):
        aligner = TemporalAligner()
        df = pd.DataFrame({"year": [2020, 2021], "value": [1000, 1100]})
        result = aligner.align_dataset(df, {"frequency": "annual"}, "test")
        assert len(result) == 24

    def test_monthly_passthrough(self):
        aligner = TemporalAligner()
        df = pd.DataFrame({
            "month_year": pd.date_range("2020-01-01", periods=12, freq="ME"),
            "value": range(12),
        })
        result = aligner.align_dataset(df, {"frequency": "monthly"}, "test")
        assert len(result) == 12
