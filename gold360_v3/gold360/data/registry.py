from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib


@dataclass
class DatasetEntry:
    name: str
    source_path: str
    data_type: str
    frequency: str
    rows: int
    columns: int
    loaded_at: str
    checksum: Optional[str] = None
    notes: Optional[str] = None
    snapshot_hash: Optional[str] = None
    snapshot_path: Optional[str] = None


@dataclass
class TransformationEntry:
    name: str
    description: str
    input_datasets: List[str]
    output_dataset: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class DataLineage:
    def __init__(self, snapshot_dir: str = "data/snapshots"):
        self.datasets: Dict[str, DatasetEntry] = {}
        self.transformations: List[TransformationEntry] = []
        self.snapshot_dir = snapshot_dir
        os.makedirs(snapshot_dir, exist_ok=True)
    
    def _hash_file(self, path: str) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]
    
    def _hash_dataframe(self, df) -> str:
        return hashlib.sha256(
            pd.util.hash_pandas_object(df, index=True).values
        ).hexdigest()[:16]
    
    def snapshot_dataframe(self, df, name: str) -> str:
        """Save a snapshot of the dataframe and return its hash."""
        import pandas as pd
        hash_val = self._hash_dataframe(df)
        snapshot_path = os.path.join(self.snapshot_dir, f"{name}_{hash_val}.parquet")
        df.to_parquet(snapshot_path, index=False)
        return hash_val, snapshot_path
    
    def register_dataset(self, name: str, path: str, data_type: str,
                         frequency: str, rows: int, columns: int,
                         checksum: Optional[str] = None,
                         snapshot_df = None) -> DatasetEntry:
        snap_hash = None
        snap_path = None
        if snapshot_df is not None:
            snap_hash, snap_path = self.snapshot_dataframe(snapshot_df, name)
        elif os.path.exists(path):
            snap_hash = self._hash_file(path)
        
        entry = DatasetEntry(
            name=name,
            source_path=path,
            data_type=data_type,
            frequency=frequency,
            rows=rows,
            columns=columns,
            loaded_at=datetime.utcnow().isoformat(),
            checksum=checksum or snap_hash,
            snapshot_hash=snap_hash,
            snapshot_path=snap_path,
        )
        self.datasets[name] = entry
        return entry
    
    def register_transformation(self, name: str, description: str,
                                 input_datasets: List[str],
                                 output_dataset: str,
                                 parameters: Optional[Dict] = None) -> TransformationEntry:
        entry = TransformationEntry(
            name=name,
            description=description,
            input_datasets=input_datasets,
            output_dataset=output_dataset,
            parameters=parameters or {},
        )
        self.transformations.append(entry)
        return entry
    
    def get_upstream(self, dataset_name: str) -> List[str]:
        upstream = set()
        for t in self.transformations:
            if t.output_dataset == dataset_name:
                upstream.update(t.input_datasets)
        return list(upstream)
    
    def lineage_report(self) -> str:
        lines = ["DATA LINEAGE REPORT", "=" * 60, ""]
        lines.append("Datasets:")
        for name, ds in sorted(self.datasets.items()):
            snap_info = f" [snap: {ds.snapshot_hash}]" if ds.snapshot_hash else ""
            lines.append(f"  {name}: {ds.data_type}/{ds.frequency}, {ds.rows}r x {ds.columns}c{snap_info}")
        lines.append("")
        lines.append("Transformations:")
        for t in self.transformations:
            lines.append(f"  {t.name}: {', '.join(t.input_datasets)} -> {t.output_dataset}")
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "datasets": {k: v.__dict__ for k, v in self.datasets.items()},
            "transformations": [t.__dict__ for t in self.transformations],
        }


import os
import pandas as pd
