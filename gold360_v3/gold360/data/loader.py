from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

from gold360.data.schemas import SCHEMA_REGISTRY, DatasetRegistry, infer_schema
from gold360.utils.config import get_project_root
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class DataLoader:
    def __init__(self, config_path: Optional[str] = None):
        self.root = get_project_root()
        if config_path is None:
            config_path = str(self.root / "config" / "data_sources.yaml")
        with open(config_path) as f:
            self.sources: Dict[str, Any] = yaml.safe_load(f)["data_sources"]
        self._cache: Dict[str, pd.DataFrame] = {}

    def list_datasets(self) -> List[str]:
        return list(self.sources.keys())

    def get_source_info(self, name: str) -> Dict[str, Any]:
        info = self.sources.get(name)
        if info is None:
            raise KeyError(f"Dataset '{name}' not found in data sources configuration")
        return info

    def load(self, name: str, use_cache: bool = True, **kwargs) -> pd.DataFrame:
        if use_cache and name in self._cache:
            logger.debug(f"Returning cached dataset '{name}'")
            return self._cache[name].copy()

        info = self.get_source_info(name)
        path = self.root / info["path"]

        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        logger.info(f"Loading dataset '{name}' from {path}")
        df = pd.read_csv(path, **kwargs)

        schema = SCHEMA_REGISTRY.get(name)
        if schema and "month" in df.columns:
            df["month"] = df["month"].astype(str)

        if use_cache:
            self._cache[name] = df.copy()

        return df

    def load_all(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        return {name: self.load(name, use_cache=use_cache) for name in self.list_datasets()}

    def inspect(self, name: str) -> Dict[str, Any]:
        df = self.load(name, use_cache=True)
        info = self.get_source_info(name)
        schema_info = infer_schema(df, name)
        return {
            "name": name,
            "source_info": info,
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "schema": schema_info,
            "missing": df.isnull().sum().to_dict(),
            "memory_mb": df.memory_usage(deep=True).sum() / 1e6,
        }

    def clear_cache(self) -> None:
        self._cache.clear()
        logger.info("DataLoader cache cleared")
