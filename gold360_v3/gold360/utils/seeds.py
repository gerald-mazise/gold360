import os
import random
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime

import numpy as np


_GLOBAL_SEED: Optional[int] = None


def set_seed(seed: Optional[int] = None) -> int:
    """Set global seed for reproducibility across all libraries."""
    global _GLOBAL_SEED
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")
    _GLOBAL_SEED = seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed


def get_seed() -> Optional[int]:
    return _GLOBAL_SEED


def hash_file(path: str) -> str:
    """Compute SHA256 hash of a file for data versioning."""
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]


def hash_dataframe(df: "pd.DataFrame") -> str:
    """Compute deterministic hash of a DataFrame."""
    import pandas as pd
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]


class RunManifest:
    """Capture full provenance for a pipeline run."""
    
    def __init__(self, run_id: Optional[str] = None, config: Optional[Dict] = None):
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = datetime.now().isoformat()
        self.seed = _GLOBAL_SEED
        self.config = config or {}
        self.data_hashes: Dict[str, str] = {}
        self.git_hash = self._get_git_hash()
        self.metrics: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
    
    def _get_git_hash(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], 
                capture_output=True, text=True, cwd=os.getcwd()
            )
            return result.stdout.strip()[:8] if result.returncode == 0 else "unknown"
        except Exception:
            return "unknown"
    
    def add_data_hash(self, name: str, path: str):
        self.data_hashes[name] = hash_file(path)
    
    def add_metrics(self, metrics: Dict[str, Any]):
        self.metrics.update(metrics)
    
    def add_artifact(self, name: str, path: str):
        self.artifacts[name] = path
    
    def save(self, output_dir: str = "reports/runs"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.run_id}.json")
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
        return path
