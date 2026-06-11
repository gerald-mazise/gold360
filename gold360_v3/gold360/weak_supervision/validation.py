from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging
from gold360.weak_supervision.labeling_functions import LabelingFunctionRegistry

logger = setup_logging(__name__)


class LFValidation:
    def __init__(self, registry: LabelingFunctionRegistry):
        self.registry = registry

    def compute_coverage(self, df: pd.DataFrame) -> Dict[str, float]:
        results = self.registry.apply_all(df)
        coverage = {}
        for name, result in results.items():
            coverage[name] = 1.0 if result.signal_strength > 0.1 else 0.0
        return coverage

    def compute_conflict_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.registry.apply_all(df)
        names = list(results.keys())
        n = len(names)
        matrix = np.zeros((n, n))

        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    si = results[name_i].signal_strength
                    sj = results[name_j].signal_strength
                    diff = abs(si - sj)
                    matrix[i][j] = 1.0 - diff

        return pd.DataFrame(matrix, index=names, columns=names)

    def compute_signal_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.registry.apply_all(df)
        rows = []
        for name, result in results.items():
            rows.append({
                "labeling_function": name,
                "signal_strength": result.signal_strength,
                "confidence": result.confidence,
            })
        return pd.DataFrame(rows)

    def report(self, df: pd.DataFrame) -> str:
        coverage = self.compute_coverage(df)
        signal = self.compute_signal_matrix(df)
        lines = ["WEAK SUPERVISION VALIDATION REPORT", "=" * 60]
        lines.append(f"\nLabeling Functions: {len(coverage)}")
        lines.append(f"\nSignal Matrix:\n{signal.to_string()}")
        lines.append(f"\nCoverage:")
        for name, cov in sorted(coverage.items()):
            lines.append(f"  {name}: {'ACTIVE' if cov > 0.5 else 'inactive'} (strength={cov:.3f})")
        return "\n".join(lines)
