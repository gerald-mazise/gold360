from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging
from gold360.utils.seeds import set_seed

logger = setup_logging(__name__)


class RobustnessTest:
    def __init__(self, n_trials: int = 20, noise_levels: List[float] = None):
        self.n_trials = n_trials
        self.noise_levels = noise_levels or [0.01, 0.05, 0.10, 0.20]
        self.results: Dict[str, Dict] = {}

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import cross_val_score, KFold

        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        baseline = cross_val_score(self._make_clf(), X, (y > 0.5).astype(int), cv=kf, scoring="roc_auc").mean()

        rows = []
        for noise_level in self.noise_levels:
            trial_scores = []
            for trial in range(self.n_trials):
                set_seed(trial)
                noise = np.random.normal(0, noise_level, X.shape)
                X_noisy = X + noise
                score = cross_val_score(self._make_clf(), X_noisy, (y > 0.5).astype(int), cv=3, scoring="roc_auc").mean()
                trial_scores.append(score)

            rows.append({
                "noise_level": noise_level,
                "mean_auc": float(np.mean(trial_scores)),
                "std_auc": float(np.std(trial_scores)),
                "degradation": float(baseline - np.mean(trial_scores)),
                "min_auc": float(np.min(trial_scores)),
                "max_auc": float(np.max(trial_scores)),
            })

        return pd.DataFrame(rows)

    def _make_clf(self):
        from catboost import CatBoostClassifier
        return CatBoostClassifier(verbose=0, random_seed=42, iterations=200, depth=4)
