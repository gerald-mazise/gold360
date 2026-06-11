from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class DataHarmonizer:
    """Merges multiple aligned datasets into a unified intelligence table."""

    def __init__(self, master_key: str = "month_year"):
        self.master_key = master_key

    def harmonize(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not datasets:
            raise ValueError("No datasets provided for harmonization")

        logger.info(f"Harmonizing {len(datasets)} datasets")

        monthly_datasets = {k: v for k, v in datasets.items() if self.master_key in v.columns}
        static_datasets = {k: v for k, v in datasets.items() if self.master_key not in v.columns}

        if not monthly_datasets:
            raise ValueError("No monthly-aligned datasets found")

        master: Optional[pd.DataFrame] = None
        for name, df in monthly_datasets.items():
            df = df.copy()
            if self.master_key in df.columns:
                df[self.master_key] = df[self.master_key].astype(str)
            if master is None:
                df = self._deduplicate_columns(df)
                master = df
                logger.debug(f"Master initialized from '{name}' ({len(df)} rows)")
            else:
                cols_to_merge = [c for c in df.columns if c != self.master_key]
                if cols_to_merge:
                    merge_df = df[[self.master_key] + cols_to_merge].copy()
                    merge_df = self._deduplicate_columns(merge_df)
                    existing_cols = set(master.columns)
                    dup_cols = [c for c in merge_df.columns if c in existing_cols and c != self.master_key]
                    if dup_cols:
                        merge_df = merge_df.rename(
                            columns={c: f"{c}_{name}" for c in dup_cols}
                        )
                    master = master.merge(merge_df, on=self.master_key, how="left")
                    logger.debug(f"Merged '{name}' -> {len(master)} rows, {len(master.columns)} cols")

        for name, df in static_datasets.items():
            logger.debug(f"Static dataset '{name}' merged as cross-join not yet supported")

        master = master.sort_values(self.master_key).reset_index(drop=True)
        nancount = master.isnull().sum().sum()
        if nancount > 0:
            logger.warning(f"{nancount} null values after harmonization")

        logger.info(f"Harmonized table: {len(master)} rows, {len(master.columns)} columns")
        return master

    def _deduplicate_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        seen: set = set()
        keep: list = []
        for col in df.columns:
            if col not in seen:
                seen.add(col)
                keep.append(col)
        return df[keep]
