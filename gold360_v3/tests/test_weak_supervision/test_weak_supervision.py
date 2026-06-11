import numpy as np
import pandas as pd
import pytest

from gold360.weak_supervision.labeling_functions import LabelingFunctionRegistry, lf_extreme_delivery_collapse
from gold360.weak_supervision.fusion import PseudoLabelFusion
from gold360.weak_supervision.validation import LFValidation
from gold360.weak_supervision.audit import LabelAuditTrail


class TestLabelingFunctionRegistry:
    def test_initialization(self):
        lf = LabelingFunctionRegistry()
        assert lf is not None

    def test_register_defaults(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        assert len(lf.list()) > 0

    def test_apply_all(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        df = pd.DataFrame({
            "delivery_gap_kg": [100, 1],
            "estimated_gold_yield_kg": [50, 200],
        })
        results = lf.apply_all(df)
        assert len(results) > 0
        assert all(name in results for name in lf.list())


class TestPseudoLabelFusion:
    def test_initialization(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        fusion = PseudoLabelFusion(lf)
        assert fusion is not None

    def test_fuse_majority_vote(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        fusion = PseudoLabelFusion(lf)
        df = pd.DataFrame({
            "delivery_gap_kg": [100, 1],
            "estimated_gold_yield_kg": [50, 200],
        })
        result = fusion.fuse(df)
        assert "pseudo_risk_probability" in result.columns
        assert "pseudo_confidence" in result.columns

    def test_fuse_weighted(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        fusion = PseudoLabelFusion(lf)
        df = pd.DataFrame({
            "delivery_gap_kg": [100, 1],
            "estimated_gold_yield_kg": [50, 200],
        })
        result = fusion.fuse(df, method="weighted")
        assert "pseudo_risk_probability" in result.columns


class TestLFValidation:
    def test_coverage(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        validator = LFValidation(lf)
        df = pd.DataFrame({
            "delivery_gap_kg": [100, 1],
            "estimated_gold_yield_kg": [50, 200],
            "official_delivery_kg": [30, 150],
        })
        covers = validator.compute_coverage(df)
        assert isinstance(covers, dict)

    def test_conflict_matrix(self):
        lf = LabelingFunctionRegistry()
        lf.register_defaults()
        validator = LFValidation(lf)
        df = pd.DataFrame({
            "delivery_gap_kg": [100, 1],
            "estimated_gold_yield_kg": [50, 200],
            "official_delivery_kg": [30, 150],
        })
        matrix = validator.compute_conflict_matrix(df)
        assert isinstance(matrix, pd.DataFrame)


class TestLabelAuditTrail:
    def test_traceability(self):
        audit = LabelAuditTrail()
        audit.record("test_lf", signal_strength=0.8, confidence=0.6, metadata={"key": "val"})
        audit.record("test_lf2", signal_strength=0.3, confidence=0.9)
        df = audit.to_dataframe()
        assert len(df) == 2
        assert "labeling_function" in df.columns
