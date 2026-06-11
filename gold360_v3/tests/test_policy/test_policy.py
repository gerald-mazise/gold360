import numpy as np
import pandas as pd
import pytest

from gold360.policy.elasticities import PolicyElasticities
from gold360.policy.engine import ScenarioEngine, ScenarioResult
from gold360.policy.scenarios import ScenarioDefinitions
from gold360.policy.overlay import PolicyOverlay


class TestPolicyElasticities:
    def test_initialization(self):
        el = PolicyElasticities()
        assert el is not None

    def test_estimate(self):
        el = PolicyElasticities()
        df = pd.DataFrame({
            "official_delivery_kg": np.random.exponential(50, 100),
            "policy_shock_flag": np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        })
        el.estimate(df)
        assert "delivery_response" in el.elasticities


class TestScenarioEngine:
    def test_initialization(self):
        import tempfile
        el = PolicyElasticities()
        engine = ScenarioEngine(el)
        assert engine is not None

    def test_simulate(self):
        el = PolicyElasticities()
        df = pd.DataFrame({
            "official_delivery_kg": np.random.exponential(50, 50),
            "policy_shock_flag": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
        })
        el.estimate(df)
        engine = ScenarioEngine(el)
        params = {"delivery_change_pct": 0.15, "residual_change_pct": -0.10}
        result = engine.simulate("test_scenario", params, df)
        assert isinstance(result, ScenarioResult)
        assert result.scenario_name == "test_scenario"


class TestScenarioDefinitions:
    def test_run_defaults(self):
        el = PolicyElasticities()
        df = pd.DataFrame({
            "official_delivery_kg": np.random.exponential(50, 50),
            "policy_shock_flag": np.random.choice([0, 1], 50, p=[0.7, 0.3]),
        })
        el.estimate(df)
        engine = ScenarioEngine(el)
        scenarios = ScenarioDefinitions(engine)
        results = scenarios.run_defaults(df)
        assert len(results) > 0
        assert "scenario" in results.columns


class TestPolicyOverlay:
    def test_compute_overlay(self):
        overlay = PolicyOverlay()
        df = pd.DataFrame({
            "risk_score": np.random.rand(50),
            "delivery_volume_kg": np.random.exponential(50, 50),
            "policy_shock_flag": np.random.choice([0, 1], 50, p=[0.9, 0.1]),
        })
        result = overlay.compute_overlay(df)
        assert result is not None
        assert any("_policy_effect" in c for c in result.columns)
