from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class TemporalValidator:
    def __init__(self, n_splits: int = 3, min_train_months: int = 12):
        self.n_splits = n_splits
        self.min_train_months = min_train_months

    def walk_forward_evaluate(self, df: pd.DataFrame, date_col: str = "month_year",
                               feature_cols: Optional[List[str]] = None,
                               target_col: str = "pseudo_risk_probability") -> List[Dict]:
        from catboost import CatBoostClassifier

        if feature_cols is None:
            feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                           if c != target_col]
        if date_col not in df.columns:
            return []

        df_sorted = df.sort_values(date_col).reset_index(drop=True)
        dates = df_sorted[date_col].unique()
        split_size = max(1, len(dates) // (self.n_splits + 1))

        results = []
        for i in range(1, self.n_splits + 1):
            train_end = i * split_size
            val_end = train_end + split_size

            train_dates = dates[:train_end]
            val_dates = dates[train_end:val_end]

            train_idx = df_sorted[date_col].isin(train_dates)
            val_idx = df_sorted[date_col].isin(val_dates)

            if train_idx.sum() < self.min_train_months or val_idx.sum() < 1:
                continue

            X_train = df_sorted.loc[train_idx, feature_cols].fillna(0).values
            y_train = df_sorted.loc[train_idx, target_col].values
            X_val = df_sorted.loc[val_idx, feature_cols].fillna(0).values
            y_val = df_sorted.loc[val_idx, target_col].values

            clf = CatBoostClassifier(verbose=0, random_seed=42, iterations=300, depth=5)
            clf.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)

            y_prob = clf.predict_proba(X_val)[:, 1] if clf.predict_proba(X_val).ndim > 1 else clf.predict_proba(X_val)
            y_bin = (y_val > 0.5).astype(int)

            from sklearn.metrics import roc_auc_score, brier_score_loss
            results.append({
                "split": i,
                "train_from": str(train_dates[0]),
                "train_to": str(train_dates[-1]),
                "val_from": str(val_dates[0]),
                "val_to": str(val_dates[-1]),
                "train_size": int(train_idx.sum()),
                "val_size": int(val_idx.sum()),
                "roc_auc": float(roc_auc_score(y_bin, y_prob)),
                "brier": float(brier_score_loss(y_bin, y_prob)),
            })

        return results

    def temporal_stability(self, df: pd.DataFrame, date_col: str = "month_year",
                            target_col: str = "pseudo_risk_probability") -> pd.DataFrame:
        if date_col not in df.columns or target_col not in df.columns:
            return pd.DataFrame()

        return df.groupby(date_col).agg(
            mean_risk=(target_col, "mean"),
            std_risk=(target_col, "std"),
            high_risk_pct=(target_col, lambda x: (x > 0.75).mean() * 100),
        ).reset_index()
