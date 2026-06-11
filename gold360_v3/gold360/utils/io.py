from pathlib import Path
from typing import Optional, Union

import pandas as pd


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_parquet(df: pd.DataFrame, path: Union[str, Path], **kwargs) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    df.to_parquet(p, index=False, **kwargs)
    return p


def load_parquet(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return pd.read_parquet(Path(path), **kwargs)
