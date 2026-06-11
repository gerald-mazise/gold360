from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class LabelAuditTrail:
    def __init__(self):
        self._entries: List[Dict[str, Any]] = []

    def record(self, lf_name: str, signal_strength: float, confidence: float,
               metadata: Optional[Dict] = None, note: Optional[str] = None):
        self._entries.append({
            "timestamp": datetime.utcnow().isoformat(),
            "labeling_function": lf_name,
            "signal_strength": signal_strength,
            "confidence": confidence,
            "metadata": metadata or {},
            "note": note or "",
        })
        logger.debug(f"Audit record: {lf_name} strength={signal_strength:.3f}")

    def record_batch(self, results: Dict[str, Any]):
        for name, result in results.items():
            self.record(
                lf_name=name,
                signal_strength=result.signal_strength,
                confidence=result.confidence,
                metadata=result.metadata,
            )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._entries)

    def summary(self) -> str:
        if not self._entries:
            return "No audit entries"
        df = self.to_dataframe()
        lines = ["LABEL AUDIT TRAIL SUMMARY", "=" * 60]
        lines.append(f"Total records: {len(df)}")
        lines.append(f"\nPer-LF Statistics:")
        for lf_name, group in df.groupby("labeling_function"):
            lines.append(
                f"  {lf_name}: mean_signal={group['signal_strength'].mean():.3f}, "
                f"mean_confidence={group['confidence'].mean():.3f}, "
                f"count={len(group)}"
            )
        lines.append(f"\nRecent Records:")
        for _, row in df.tail(5).iterrows():
            lines.append(
                f"  [{row['timestamp'][:19]}] {row['labeling_function']}: "
                f"signal={row['signal_strength']:.3f}, "
                f"conf={row['confidence']:.3f}"
            )
        return "\n".join(lines)
