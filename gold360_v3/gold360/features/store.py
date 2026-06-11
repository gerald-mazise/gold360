from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from gold360.utils.config import get_project_root
from gold360.utils.io import ensure_dir, load_parquet, save_parquet
from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class FeatureStore:
    def __init__(self, version: str = "v001"):
        self.root = get_project_root()
        self.store_dir = ensure_dir(self.root / "data" / "feature_store")
        self.version = version

    def _feature_path(self, name: str) -> Path:
        return self.store_dir / f"{name}_{self.version}.parquet"

    def _metadata_path(self) -> Path:
        return self.store_dir / f"metadata_{self.version}.parquet"

    def write_features(self, name: str, df: pd.DataFrame, metadata: Optional[Dict] = None) -> Path:
        path = self._feature_path(name)
        save_parquet(df, path)
        logger.info(f"Wrote feature '{name}' ({len(df)} rows) to {path}")
        if metadata:
            self._write_metadata(name, metadata)
        return path

    def read_features(self, name: str) -> pd.DataFrame:
        path = self._feature_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Feature '{name}' not found at {path}")
        df = load_parquet(path)
        logger.debug(f"Read feature '{name}' ({len(df)} rows)")
        return df

    def list_features(self) -> List[str]:
        return sorted(set(
            p.stem.replace(f"_{self.version}", "")
            for p in self.store_dir.glob(f"*_{self.version}.parquet")
        ))

    def _write_metadata(self, name: str, metadata: Dict) -> None:
        meta_path = self._metadata_path()
        try:
            existing = load_parquet(meta_path)
        except (FileNotFoundError, FileNotFoundError):
            existing = pd.DataFrame(columns=["feature", "version", "timestamp", "rows", "cols"])
        new_row = pd.DataFrame([{
            "feature": name,
            "version": self.version,
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "rows": metadata.get("rows", 0),
            "cols": metadata.get("cols", 0),
        }])
        updated = pd.concat([existing, new_row], ignore_index=True)
        save_parquet(updated, meta_path)

    def metadata_report(self) -> str:
        try:
            df = load_parquet(self._metadata_path())
            lines = ["FEATURE STORE METADATA", "=" * 60]
            for _, row in df.iterrows():
                lines.append(
                    f"  {row['feature']}: v{row['version']}, "
                    f"{int(row['rows'])}r x {int(row['cols'])}c @ {row['timestamp']}"
                )
            return "\n".join(lines)
        except FileNotFoundError:
            return "No metadata found"
