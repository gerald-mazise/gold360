from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.fusion.layer import FusionLayer, IntelligenceTensor
from gold360.models.classifier import CatBoostClassifier
from gold360.models.trainer import ModelTrainer
from gold360.utils.config import get_project_root
from gold360.utils.logging import setup_logging
from gold360.utils.seeds import set_seed

logger = setup_logging(__name__)


class TrainingPipeline:
    def __init__(self, config: Optional[Dict] = None):
        self.root = get_project_root()
        self.config = config or {}
        self.trainer = ModelTrainer(self.config)
        self.fusion = FusionLayer()

    def train_from_tensor(self, tensor: IntelligenceTensor) -> CatBoostClassifier:
        X = tensor.get_X()
        y = tensor.get_y()
        feature_names = tensor.get_feature_names()

        df = tensor.df.copy()
        df["__target__"] = y
        for i, name in enumerate(feature_names):
            df[name] = X[:, i]

        train, val, test = self.trainer.temporal_split(df)
        y_train = train["__target__"].values
        X_train = train[feature_names].fillna(0).values
        X_val = val[feature_names].fillna(0).values if len(val) > 0 else None
        y_val = val["__target__"].values if len(val) > 0 else None

        clf = self.trainer.train(X_train, y_train, X_val, y_val)

        model_dir = self.root / "models"
        model_dir.mkdir(exist_ok=True)
        clf.save(model_dir / "gold360_catboost_trained.joblib")

        cv_metrics = self.trainer.cross_validate(X, y)

        logger.info(f"Training complete. CV metrics: {cv_metrics}")
        return clf
