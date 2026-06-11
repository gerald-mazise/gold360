from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import mlflow
    import mlflow.catboost
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from gold360.anomaly.ensemble import AnomalyEnsemble
from gold360.data.harmonizer import DataHarmonizer
from gold360.data.loader import DataLoader
from gold360.data.registry import DataLineage
from gold360.data.temporal import TemporalAligner
from gold360.data.validator import DataValidator
from gold360.explainability.shap_explainer import SHAPExplainer
from gold360.explainability.report import ExplanationReport
from gold360.features.delivery import DeliveryFeatures
from gold360.features.governance import GovernanceFeatures
from gold360.features.macro import MacroFeatures
from gold360.features.operational import OperationalFeatures
from gold360.features.registry import FeatureRegistry
from gold360.features.spatial import SpatialFeatures
from gold360.features.store import FeatureStore
from gold360.features.trade import TradeFeatures
from gold360.fusion.layer import FusionLayer
from gold360.fusion.tensor import IntelligenceTensor
from gold360.models.predict import Predictor
from gold360.models.trainer import ModelTrainer
from gold360.policy.elasticities import PolicyElasticities
from gold360.policy.engine import ScenarioEngine
from gold360.policy.scenarios import ScenarioDefinitions
from gold360.utils.config import get_project_root
from gold360.utils.logging import setup_logging
from gold360.utils.seeds import set_seed, RunManifest, get_seed
from gold360.weak_supervision.fusion import PseudoLabelFusion
from gold360.weak_supervision.labeling_functions import LabelingFunctionRegistry

logger = setup_logging(__name__)


class PipelineOrchestrator:
    def __init__(self, config: Optional[Dict] = None):
        self.root = get_project_root()
        self.config = config or {}
        self.lineage = DataLineage()

        self.loader = DataLoader()
        self.validator = DataValidator()
        self.aligner = TemporalAligner()
        self.harmonizer = DataHarmonizer()
        self.feature_registry = FeatureRegistry()

        self.delivery_features = DeliveryFeatures(self.feature_registry)
        self.macro_features = MacroFeatures(self.feature_registry)
        self.operational_features = OperationalFeatures(self.feature_registry)
        self.governance_features = GovernanceFeatures(self.feature_registry)
        self.spatial_features = SpatialFeatures(self.feature_registry)
        self.trade_features = TradeFeatures(self.feature_registry)

        self.lf_registry = LabelingFunctionRegistry()
        self.lf_registry.register_defaults()
        self.pseudo_fusion = PseudoLabelFusion(self.lf_registry)
        self.anomaly_ensemble = AnomalyEnsemble()
        self.fusion_layer = FusionLayer()
        self.model_trainer = ModelTrainer(self.config)
        self.feature_store = FeatureStore()

    def run_full(self, seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            set_seed(seed)
        
        # Initialize run manifest for reproducibility
        manifest = RunManifest(config=self.config)

        mlflow_run = None
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_tracking_uri(f"sqlite:///{self.root}/mlflow.db")
                mlflow_run = mlflow.start_run(run_name=f"gold360_{manifest.run_id[:8]}")
                mlflow.log_param("run_id", manifest.run_id)
                mlflow.log_param("seed", get_seed())
                mlflow.log_params(self.config)
            except Exception as e:
                logger.warning(f"MLflow init failed: {e}")
                mlflow_run = None
        
        logger.info("=" * 60)
        logger.info("GOLD360 FULL PIPELINE START")
        logger.info("=" * 60)

        datasets = self.loader.load_all()
        logger.info(f"Loaded {len(datasets)} datasets")

        for name, df in datasets.items():
            self.validator.validate(df, name)
            info = self.loader.get_source_info(name)
            # Register with data snapshot for lineage
            self.lineage.register_dataset(
                name=name, path=info["path"], data_type=info["type"],
                frequency=info["frequency"], rows=len(df), columns=len(df.columns),
                snapshot_df=df
            )
            manifest.add_data_hash(name, info["path"])

        aligned = {}
        for name, df in datasets.items():
            info = self.loader.get_source_info(name)
            aligned[name] = self.aligner.align_dataset(df, info, name)

        unified = self.harmonizer.harmonize(aligned)
        self.lineage.register_transformation(
            "harmonization", "Multi-source temporal harmonization",
            list(aligned.keys()), "unified_intelligence",
        )

        feature_groups = [
            ("delivery", self.delivery_features.registry),
            ("macro", self.macro_features.registry),
            ("operational", self.operational_features.registry),
            ("governance", self.governance_features.registry),
            ("spatial", self.spatial_features.registry),
            ("trade", self.trade_features.registry),
        ]

        for group_name, registry in feature_groups:
            group_features = registry.list_features(group=group_name)
            for feat in group_features:
                try:
                    result = registry.compute(feat.name, unified)
                    if isinstance(result, pd.Series):
                        unified[feat.name] = result.values if len(result) == len(unified) else result
                    else:
                        unified[feat.name] = result
                except Exception as e:
                    logger.warning(f"Feature '{feat.name}' computation failed: {e}")
                    unified[feat.name] = 0.0

        pseudo_labeled = self.pseudo_fusion.fuse(unified)
        pl = pseudo_labeled[["pseudo_risk_probability", "pseudo_confidence", "num_active_lfs"]].iloc[0:1]

        anomaly_features = [c for c in unified.columns if unified[c].dtype in (np.float64, np.int64)]
        unified[anomaly_features] = unified[anomaly_features].fillna(0)
        anomaly_result = self.anomaly_ensemble.fit(unified, anomaly_features).predict(unified)

        fused = self.fusion_layer.fuse(unified, pl, anomaly_result)

        feature_cols = [
            c for c in fused.columns
            if c not in ("month_year", "mine_id", "province", "mine_name",
                         "pseudo_risk_probability", "pseudo_confidence")
            and fused[c].dtype in (np.float64, np.int64, np.float32)
        ]

        target_col = "pseudo_risk_probability"
        if target_col in fused.columns:
            y = fused[target_col].values
        else:
            logger.warning("No pseudo-label target found; using synthetic risk_flag")
            y = fused.get("risk_flag", pd.Series(0.0, index=fused.index)).values

        tensor = IntelligenceTensor(fused)
        tensor.set_features(feature_cols).set_target(target_col if target_col in fused.columns else "risk_flag")
        X = tensor.get_X()
        y = tensor.get_y()

        train, val, test = self.model_trainer.temporal_split(fused)
        X_train = tensor.df.loc[train.index, feature_cols].fillna(0).values
        y_train = train[target_col].values if target_col in train.columns else train.get("risk_flag", pd.Series(0.0, index=train.index)).values
        X_val = tensor.df.loc[val.index, feature_cols].fillna(0).values if len(val) > 0 else None
        y_val = val[target_col].values if len(val) > 0 and target_col in val.columns else None
        X_test = tensor.df.loc[test.index, feature_cols].fillna(0).values if len(test) > 0 else None

        clf = self.model_trainer.train(X_train, y_train, X_val, y_val)

        model_dir = self.root / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / "gold360_catboost_v001.joblib"
        clf.save(model_path)

        # Log model to MLflow
        if mlflow_run is not None:
            try:
                mlflow.log_param("n_features", len(feature_cols))
                mlflow.log_param("train_samples", len(train))
                mlflow.log_param("feature_groups", [g[0] for g in feature_groups])
                mlflow.log_artifact(str(model_path))
            except Exception as e:
                logger.warning(f"MLflow model logging failed: {e}")

        predictor = Predictor(clf)
        test_predictions = None
        if X_test is not None and len(test) > 0:
            test_meta = test[["month_year"]].copy() if "month_year" in test.columns else None
            test_predictions = predictor.predict_with_confidence(X_test, feature_cols, test_meta)

        elasticities = PolicyElasticities()
        elasticities.estimate(fused)

        scenario_engine = ScenarioEngine(elasticities)
        scenarios = ScenarioDefinitions(scenario_engine)
        scenario_results = scenarios.run_defaults(fused)

        results = {
            "datasets": {k: len(v) for k, v in datasets.items()},
            "validation": {k: v.summary() for k, v in self.validator.results.items()},
            "features": len(feature_cols),
            "anomaly_flagged": int((anomaly_result["anomaly_consensus_score"] > 50).sum()),
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "scenarios": scenario_results.to_dict("records") if len(scenario_results) > 0 else [],
            "model_dir": str(model_dir),
            "run_manifest": manifest.run_id,
        }
        
        # Save run manifest with metrics
        manifest.add_metrics(results)
        manifest.add_artifact("model", str(model_path))
        manifest.save()

        # Final MLflow logging
        if mlflow_run is not None:
            try:
                mlflow.log_metrics({
                    "train_samples": len(train),
                    "val_samples": len(val),
                    "test_samples": len(test),
                    "n_features": len(feature_cols),
                    "anomaly_flagged": results.get("anomaly_flagged", 0),
                    "n_scenarios": len(results.get("scenarios", [])),
                })
                for artifact in manifest.artifacts:
                    mlflow.log_artifact(artifact)
            except Exception as e:
                logger.warning(f"MLflow final logging failed: {e}")
            finally:
                mlflow.end_run()

        logger.info("=" * 60)
        logger.info("GOLD360 FULL PIPELINE COMPLETE")
        logger.info("=" * 60)

        return results
