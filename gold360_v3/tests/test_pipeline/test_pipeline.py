import numpy as np
import pandas as pd
import pytest

from gold360.pipeline.orchestrator import PipelineOrchestrator


class TestPipelineOrchestrator:
    def test_initialization(self):
        orchestrator = PipelineOrchestrator()
        assert orchestrator is not None
        assert orchestrator.loader is not None
        assert orchestrator.harmonizer is not None

    def test_component_availability(self):
        orchestrator = PipelineOrchestrator()
        assert hasattr(orchestrator, "delivery_features")
        assert hasattr(orchestrator, "macro_features")
        assert hasattr(orchestrator, "operational_features")
        assert hasattr(orchestrator, "anomaly_ensemble")
        assert hasattr(orchestrator, "pseudo_fusion")
        assert hasattr(orchestrator, "fusion_layer")
        assert hasattr(orchestrator, "model_trainer")
