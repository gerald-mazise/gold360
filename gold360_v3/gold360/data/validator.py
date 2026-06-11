from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class ValidationResult:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.checks: List[Dict[str, Any]] = []
        self.passed: int = 0
        self.failed: int = 0
        self.warnings: int = 0

    def add_check(self, name: str, status: str, detail: Optional[str] = None):
        self.checks.append({"check": name, "status": status, "detail": detail})
        if status == "PASS":
            self.passed += 1
        elif status == "FAIL":
            self.failed += 1
        else:
            self.warnings += 1

    @property
    def is_valid(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        return (
            f"[{self.dataset_name}] {self.passed} passed, "
            f"{self.failed} failed, {self.warnings} warnings"
        )


class DataValidator:
    def __init__(self):
        self.results: Dict[str, ValidationResult] = {}

    def validate(self, df: pd.DataFrame, name: str, checks: Optional[List[str]] = None) -> ValidationResult:
        result = ValidationResult(name)
        self.results[name] = result

        default_checks = ["missing", "temporal", "duplicates", "schema", "outliers"]
        for check_name in checks or default_checks:
            check_fn = getattr(self, f"_check_{check_name}", None)
            if check_fn:
                try:
                    check_fn(df, name, result)
                except Exception as e:
                    result.add_check(check_name, "FAIL", str(e))
            else:
                result.add_check(check_name, "WARN", f"No validator for check '{check_name}'")

        logger.info(result.summary())
        return result

    def _check_missing(self, df: pd.DataFrame, name: str, result: ValidationResult):
        missing_cols = df.columns[df.isnull().any()].tolist()
        if not missing_cols:
            result.add_check("missing_values", "PASS", "No missing values")
            return
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        pct = (missing_cells / total_cells) * 100
        if pct < 1:
            result.add_check("missing_values", "PASS", f"{pct:.2f}% missing across {missing_cols}")
        elif pct < 5:
            result.add_check("missing_values", "WARN", f"{pct:.2f}% missing across {missing_cols}")
        else:
            result.add_check("missing_values", "FAIL", f"{pct:.2f}% missing across {missing_cols}")

    def _check_temporal(self, df: pd.DataFrame, name: str, result: ValidationResult):
        date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "month", "period", "year"])]
        if not date_cols:
            result.add_check("temporal_integrity", "WARN", "No date column found")
            return
        result.add_check("temporal_integrity", "PASS", f"Date columns: {date_cols}")

    def _check_duplicates(self, df: pd.DataFrame, name: str, result: ValidationResult):
        if df.duplicated().sum() == 0:
            result.add_check("duplicates", "PASS", "No duplicate rows")
        else:
            dup_count = df.duplicated().sum()
            result.add_check("duplicates", "FAIL", f"{dup_count} duplicate rows found")

    def _check_schema(self, df: pd.DataFrame, name: str, result: ValidationResult):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            inf_mask = np.isinf(df[col])
            if inf_mask.any():
                result.add_check(
                    "schema_consistency", "WARN",
                    f"Column '{col}' has {inf_mask.sum()} infinite values"
                )
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        result.add_check("schema_consistency", "PASS", f"{len(numeric_cols)} numeric columns ok")

    def _check_outliers(self, df: pd.DataFrame, name: str, result: ValidationResult):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_count = 0
        for col in numeric_cols:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_count += ((df[col] < lower) | (df[col] > upper)).sum()
        total = df[numeric_cols].size if numeric_cols else 1
        pct = (outlier_count / max(total, 1)) * 100
        result.add_check("outlier_detection", "PASS" if pct < 10 else "WARN",
                         f"{pct:.1f}% outlier rate")

    def full_report(self) -> str:
        lines = ["=" * 60, "DATA VALIDATION REPORT", "=" * 60]
        for name, result in self.results.items():
            lines.append("")
            lines.append(f"Dataset: {name}")
            lines.append("-" * 40)
            for check in result.checks:
                lines.append(f"  [{check['status']}] {check['check']}: {check.get('detail', '')}")
            lines.append(f"  -> {result.summary()}")
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)
