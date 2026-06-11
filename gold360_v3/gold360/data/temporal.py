from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class TemporalAligner:
    """Aligns multi-frequency datasets to monthly temporal granularity."""

    def __init__(self, master_key: str = "month_year"):
        self.master_key = master_key
        self._aligned: Dict[str, pd.DataFrame] = {}

    def _period_to_month_start(self, period: str) -> str:
        if "-Q" in period:
            year, q = period.split("-Q")
            month = str((int(q) - 1) * 3 + 1).zfill(2)
            return f"{year}-{month}"
        return period

    def _expand_quarterly(self, df: pd.DataFrame, period_col: str) -> pd.DataFrame:
        records = []
        for _, row in df.iterrows():
            period = str(row[period_col])
            if "Q" not in period or pd.isna(row[period_col]):
                continue
            try:
                year, q = period.split("-Q")
                yr = int(float(year))
                base_month = (int(q) - 1) * 3 + 1
            except (ValueError, TypeError):
                continue
            for offset in range(3):
                month = base_month + offset
                if month > 12:
                    month -= 12
                month_str = f"{yr}-{str(month).zfill(2)}"
                r = row.to_dict()
                r[self.master_key] = month_str
                r.pop(period_col, None)
                records.append(r)
        result = pd.DataFrame(records)
        logger.debug(f"Expanded quarterly data: {len(df)} -> {len(result)} rows")
        return result

    def _expand_annual(self, df: pd.DataFrame, year_col: str = "year") -> pd.DataFrame:
        df = df.copy()
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=[year_col])
        records = []
        for _, row in df.iterrows():
            yr = int(row[year_col])
            for month in range(1, 13):
                month_str = f"{yr}-{str(month).zfill(2)}"
                r = row.to_dict()
                r[self.master_key] = month_str
                r.pop(year_col, None)
                records.append(r)
        result = pd.DataFrame(records)
        logger.debug(f"Expanded annual data: {len(df)} -> {len(result)} rows")
        return result

    def _align_column_types(self, df: pd.DataFrame) -> pd.DataFrame:
        text_cols = {self.master_key, "period", "quarter", "year", "period_col", "month", "month_year"}
        for col in df.columns:
            if col in text_cols:
                df[col] = df[col].astype(str)
            elif col == self.master_key:
                df[col] = df[col].astype(str)
            elif df[col].dtype == object:
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
                except (ValueError, TypeError):
                    pass
        return df

    def align(self, df: pd.DataFrame, frequency: str, period_col: Optional[str] = None, name: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df = self._align_column_types(df)

        if frequency == "monthly":
            if period_col and period_col in df.columns:
                df = df.rename(columns={period_col: self.master_key})
            elif self.master_key not in df.columns:
                raise ValueError(f"Monthly data must have '{self.master_key}' column")

        elif frequency == "quarterly":
            pc = period_col or "period"
            df = self._expand_quarterly(df, pc)

        elif frequency == "annual":
            yc = period_col or "year"
            df = self._expand_annual(df, yc)

        elif frequency == "static":
            df[self.master_key] = "2020-01"

        else:
            logger.warning(f"Unknown frequency '{frequency}', attempting passthrough")
            if self.master_key not in df.columns:
                df[self.master_key] = "2020-01"

        if name:
            self._aligned[name] = df.copy()
            logger.info(f"Aligned '{name}' ({frequency}) -> {len(df)} rows")

        return df

    def align_dataset(self, df: pd.DataFrame, source_info: dict, name: str) -> pd.DataFrame:
        freq = source_info.get("frequency", "monthly")
        period_col = None
        for col in df.columns:
            if col in ("period", "date", "month", "year_month", "month_year", "year"):
                period_col = col
                break
        return self.align(df, freq, period_col=period_col, name=name)
