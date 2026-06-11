from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)

APPROVED_LANGUAGE = {
    "positive": "associated with elevated",
    "negative": "associated with reduced",
    "high_risk": "elevated delivery shortfall risk",
    "low_risk": "reduced delivery shortfall risk",
}


class ExplanationReport:
    @staticmethod
    def generate(waterfall_data: Dict, top_n: int = 5) -> str:
        features = waterfall_data["features"][:top_n]
        parts = []

        expected = waterfall_data["expected_value"]
        actual = waterfall_data["output_value"]
        diff = actual - expected

        if diff > 0:
            parts.append(f"Risk assessment: {APPROVED_LANGUAGE['high_risk']} "
                         f"(model output deviates +{diff:.3f} from baseline)")
        else:
            parts.append(f"Risk assessment: {APPROVED_LANGUAGE['low_risk']} "
                         f"(model output deviates {diff:.3f} from baseline)")

        parts.append("\nPrimary contributing factors:")

        for i, feat in enumerate(features, 1):
            direction = "increases" if feat["shap_value"] > 0 else "decreases"
            strength = "strongly" if abs(feat["shap_value"]) > 0.1 else "moderately"
            if feat["shap_value"] > 0:
                lang = APPROVED_LANGUAGE["positive"]
            else:
                lang = APPROVED_LANGUAGE["negative"]
            parts.append(
                f"  {i}. {feat['name']} ({feat['value']:.2f}) "
                f"{strength} {lang} risk "
                f"(SHAP: {feat['shap_value']:+.4f})"
            )

        parts.append("\nInterpretation note: This is a probabilistic anomaly signal, "
                     "not a determination of illicit activity.")
        return "\n".join(parts)

    @staticmethod
    def generate_batch(waterfall_list: List[Dict]) -> List[str]:
        return [ExplanationReport.generate(w) for w in waterfall_list]
