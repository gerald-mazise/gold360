from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


@dataclass
class LabelingFunctionResult:
    name: str
    signal_strength: float
    confidence: float
    metadata: Dict = field(default_factory=dict)


LabelingFunction = Callable[[pd.DataFrame], LabelingFunctionResult]


def lf_extreme_delivery_collapse(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if "delivery_gap_kg" in df.columns and "estimated_gold_yield_kg" in df.columns:
        gap_ratio = df["delivery_gap_kg"] / df["estimated_gold_yield_kg"].replace(0, np.nan)
        recent_gap = gap_ratio.tail(3).mean() if len(df) >= 3 else gap_ratio.mean()
        score = float(np.clip(recent_gap * 2, 0, 1) if not np.isnan(recent_gap) else 0)
        gap_mean = float(gap_ratio.mean()) if not np.isnan(gap_ratio.mean()) else 0
        gap_std = float(gap_ratio.std()) if not np.isnan(gap_ratio.std()) else 0
        if gap_std > 0:
            z = (recent_gap - gap_mean) / gap_std if not np.isnan(recent_gap) else 0
            confidence = float(np.clip(abs(z) / 4, 0, 1))
        meta["gap_ratio_mean"] = gap_mean
        meta["gap_ratio_latest"] = float(recent_gap) if not np.isnan(recent_gap) else 0

    return LabelingFunctionResult(
        name="extreme_delivery_collapse",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_fx_arbitrage_stress(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if "fx_market_rate" in df.columns and "official_delivery_kg" in df.columns:
        fx = df["fx_market_rate"]
        fx_z = ((fx - fx.mean()) / fx.std()).tail(3).mean() if len(df) >= 3 and fx.std() > 0 else 0
        delivery_change = df["official_delivery_kg"].pct_change().tail(3).mean() if len(df) >= 3 else 0
        if not np.isnan(fx_z) and not np.isnan(delivery_change):
            score = float(np.clip((float(fx_z) * 0.2 - float(delivery_change) * 0.8), 0, 1))
            confidence = float(np.clip(abs(score) * 1.5, 0.1, 0.9))
        meta["fx_z_latest"] = float(fx_z) if not np.isnan(fx_z) else 0
        meta["delivery_change"] = float(delivery_change) if not np.isnan(delivery_change) else 0

    return LabelingFunctionResult(
        name="fx_arbitrage_stress",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_impossible_yield_contradiction(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if all(c in df.columns for c in ["estimated_gold_yield_kg", "official_delivery_kg"]):
        contradiction = (df["estimated_gold_yield_kg"] > df["estimated_gold_yield_kg"].quantile(0.75)) & \
                        (df["official_delivery_kg"] < df["official_delivery_kg"].quantile(0.25))
        contradiction_rate = contradiction.tail(6).mean() if len(df) >= 6 else contradiction.mean()
        score = float(np.clip(contradiction_rate * 3, 0, 1))
        confidence = float(np.clip(contradiction_rate * 2, 0.1, 0.85))
        meta["contradiction_rate"] = float(contradiction_rate)

    return LabelingFunctionResult(
        name="impossible_yield_contradiction",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_corridor_inconsistency(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if all(c in df.columns for c in ["distance_to_border_km", "delivery_gap_kg"]):
        near_border = df["distance_to_border_km"] < 100
        gap_high = df["delivery_gap_kg"] > df["delivery_gap_kg"].median()
        inconsistency = (near_border & gap_high).mean()
        score = float(np.clip(inconsistency * 2, 0, 1))
        confidence = float(np.clip(inconsistency * 1.5, 0.1, 0.8))
        meta["inconsistency_rate"] = float(inconsistency)

    return LabelingFunctionResult(
        name="corridor_inconsistency",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_inventory_anomaly(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if "inventory_carryover_kg" in df.columns:
        inventory = df["inventory_carryover_kg"]
        persistent = (inventory > inventory.median()).mean()
        score = float(np.clip(persistent * 1.5, 0, 1))
        confidence = float(np.clip(persistent, 0.1, 0.7))
        meta["persistent_inventory_rate"] = float(persistent)

    return LabelingFunctionResult(
        name="inventory_anomaly",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_policy_contradiction(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    if all(c in df.columns for c in ["policy_shock_flag", "official_delivery_kg"]):
        post_policy = df[df["policy_shock_flag"] > 0]
        if len(post_policy) >= 2:
            delivery_after = post_policy["official_delivery_kg"].mean()
            delivery_before = df[df["policy_shock_flag"] == 0]["official_delivery_kg"].mean()
            if delivery_before > 0:
                drop = max(0, (delivery_before - delivery_after) / delivery_before)
                score = float(np.clip(drop * 2, 0, 1))
                confidence = float(np.clip(drop * 1.5, 0.1, 0.8))
                meta["delivery_drop_after_policy"] = float(drop)

    return LabelingFunctionResult(
        name="policy_contradiction",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


def lf_operational_mismatch(df: pd.DataFrame) -> LabelingFunctionResult:
    score = 0.0
    confidence = 0.0
    meta = {}

    has_rain = "rainfall_mm" in df.columns
    has_energy = "energy_consumption" in df.columns
    has_delivery = "official_delivery_kg" in df.columns

    if has_rain and has_delivery:
        if df["rainfall_mm"].std() > 0:
            rain_z = abs((df["rainfall_mm"] - df["rainfall_mm"].mean()) / df["rainfall_mm"].std())
            high_rain = rain_z > 1.5
            delivery_unexpected = df["official_delivery_kg"] > df["official_delivery_kg"].quantile(0.75)
            mismatch = (high_rain & delivery_unexpected).mean()
            score += float(mismatch * 0.5)

    if has_energy and has_delivery:
        if df["energy_consumption"].std() > 0:
            energy_z = abs((df["energy_consumption"] - df["energy_consumption"].mean()) / df["energy_consumption"].std())
            low_energy = energy_z < -0.5
            delivery_high = df["official_delivery_kg"] > df["official_delivery_kg"].quantile(0.75)
            mismatch2 = (low_energy & delivery_high).mean()
            score += float(mismatch2 * 0.5)

    score = float(np.clip(score, 0, 1))
    confidence = float(np.clip(score * 0.7, 0.1, 0.75))
    meta["composite_mismatch"] = score

    return LabelingFunctionResult(
        name="operational_mismatch",
        signal_strength=score,
        confidence=confidence,
        metadata=meta,
    )


DEFAULT_LFS: Dict[str, LabelingFunction] = {
    "extreme_delivery_collapse": lf_extreme_delivery_collapse,
    "fx_arbitrage_stress": lf_fx_arbitrage_stress,
    "impossible_yield_contradiction": lf_impossible_yield_contradiction,
    "corridor_inconsistency": lf_corridor_inconsistency,
    "inventory_anomaly": lf_inventory_anomaly,
    "policy_contradiction": lf_policy_contradiction,
    "operational_mismatch": lf_operational_mismatch,
}


class LabelingFunctionRegistry:
    def __init__(self):
        self._lfs: Dict[str, LabelingFunction] = {}

    def register(self, name: str, fn: LabelingFunction):
        self._lfs[name] = fn
        logger.debug(f"Registered labeling function: {name}")

    def register_defaults(self):
        for name, fn in DEFAULT_LFS.items():
            self.register(name, fn)

    def get(self, name: str) -> LabelingFunction:
        if name not in self._lfs:
            raise KeyError(f"Labeling function '{name}' not registered")
        return self._lfs[name]

    def list(self) -> List[str]:
        return list(self._lfs.keys())

    def apply_all(self, df: pd.DataFrame) -> Dict[str, LabelingFunctionResult]:
        results = {}
        for name, fn in self._lfs.items():
            try:
                results[name] = fn(df)
            except Exception as e:
                logger.warning(f"LF '{name}' failed: {e}")
                results[name] = LabelingFunctionResult(
                    name=name, signal_strength=0.0, confidence=0.0,
                    metadata={"error": str(e)}
                )
        return results

    def apply_all_rowwise(self, df: pd.DataFrame) -> pd.DataFrame:
        results = self.apply_all(df)
        rows = []
        for lf_name, result in results.items():
            rows.append({
                "labeling_function": lf_name,
                "signal_strength": result.signal_strength,
                "confidence": result.confidence,
                **{f"meta_{k}": v for k, v in result.metadata.items()},
            })
        return pd.DataFrame(rows)
