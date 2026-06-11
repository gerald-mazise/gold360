from datetime import date
from enum import Enum
from typing import Dict, List, Optional, Type

import pandas as pd
from pydantic import BaseModel, Field, field_validator


class MinerType(str, Enum):
    ASM = "ASM"
    LSM = "LSM"


class LicenseStatus(str, Enum):
    LICENSED = "Licensed"
    COOPERATIVE = "Cooperative"
    INFORMAL = "Informal"
    PENDING = "Pending"


class RiskCategory(str, Enum):
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"


class VerificationStatus(str, Enum):
    VERIFIED = "Verified"


class PolicyDirection(str, Enum):
    WIDENING = "widening"
    NARROWING = "narrowing"
    NEUTRAL = "neutral"


class PolicyLag(str, Enum):
    IMMEDIATE = "immediate"
    SHORT = "1-3 months"
    MEDIUM = "3-6 months"


class FGRDeliveryRecord(BaseModel):
    period: str
    small_scale_producers_kg: float
    primary_producers_kg: float
    total_fgr_deliveries_kg: float
    source: str
    verification_status: VerificationStatus

    @field_validator("period")
    @classmethod
    def validate_period_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-Q[1-4]$", v):
            raise ValueError(f"Period must be YYYY-Qn format, got {v}")
        return v


class ZimstatProductionRecord(BaseModel):
    period: str
    zimstat_production_kg: float

    @field_validator("period")
    @classmethod
    def validate_period_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-Q[1-4]$", v):
            raise ValueError(f"Period must be YYYY-Qn format, got {v}")
        return v


class SyntheticMineOpsRecord(BaseModel):
    month: str
    mine_id: str
    mine_name: str
    province: str
    district: str
    miner_type: MinerType
    license_status: LicenseStatus
    mine_latitude: float
    mine_longitude: float
    nearest_fidelity_office: str
    distance_to_fidelity_km: float
    nearest_border_post: str
    distance_to_border_km: float
    ore_processed_tonnes: float
    ore_grade_gpt: float
    recovery_rate_pct: float
    estimated_gold_yield_kg: float
    official_delivery_kg: float
    delivery_gap_kg: float
    payment_delay_days: int
    gold_price_usd: float
    fx_market_rate: float
    inflation_rate: float
    rainfall_mm: float
    national_gap_ratio: float
    mirror_trade_gap_ratio: float
    policy_shock_flag: int

    @field_validator("month")
    @classmethod
    def validate_month_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"Month must be YYYY-MM format, got {v}")
        return v


class GoldPriceRecord(BaseModel):
    date: str
    gold_price_usd: float

    @field_validator("date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"Date must be YYYY-MM format, got {v}")
        return v


class FXMarketRecord(BaseModel):
    year: int
    official_exchange_rate: float
    parallel_exchange_rate: float
    parallel_premium_ratio: float
    fx_distortion_score: int


class InflationRecord(BaseModel):
    year: int
    usd_exchange_rate: float


class CPIRecord(BaseModel):
    period: int
    cpi_index: float


class RainfallRecord(BaseModel):
    year_month: str
    province: str
    monthly_rainfall_mm: float

    @field_validator("year_month")
    @classmethod
    def validate_month_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"year_month must be YYYY-MM format, got {v}")
        return v


class EnergyRecord(BaseModel):
    year: int
    primary_energy_consumption: float


class PolicyEventRecord(BaseModel):
    month_year: str
    scope: str
    event_type: str
    institution_or_region: str
    channel_affected: str
    sector_target: str
    direction: str
    expected_lag: str
    description: str

    @field_validator("month_year")
    @classmethod
    def validate_month_format(cls, v: str) -> str:
        import re
        if not re.match(r"^\d{4}-\d{2}$", v):
            raise ValueError(f"month_year must be YYYY-MM format, got {v}")
        return v


class FGROfficeRecord(BaseModel):
    office_name: str
    classification: str
    province: str
    district: str
    physical_address: str
    operational_host: str
    latitude: float
    longitude: float


class MirrorTradeRecord(BaseModel):
    year: int
    zimbabwe_exports_to_uae_usd_hs710813: Optional[float] = None
    zimbabwe_exports_kg: Optional[float] = None
    uae_imports_from_zimbabwe_usd_hs710812: Optional[float] = None
    uae_imports_kg: Optional[float] = None
    oec_mirror_estimate_usd: Optional[float] = None
    source_notes: Optional[str] = None


class GoldExportRecord(BaseModel):
    year: int
    gold_export_value_usd: float
    mass_kg: float
    data_status: str
    source: str


class SmugglingIncidentRecord(BaseModel):
    incident_date: str
    suspects_affiliation: str
    intercept_location: str
    intercept_country: str
    outside_zimbabwe_flag: str
    quantity_kg: float
    estimated_value_usd: str
    operational_details: str


SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "fgr_deliveries": FGRDeliveryRecord,
    "zimstat_production": ZimstatProductionRecord,
    "synthetic_mine_ops": SyntheticMineOpsRecord,
    "gold_price": GoldPriceRecord,
    "fx_market": FXMarketRecord,
    "inflation": InflationRecord,
    "cpi": CPIRecord,
    "rainfall": RainfallRecord,
    "energy": EnergyRecord,
    "policy_events": PolicyEventRecord,
    "fgr_offices": FGROfficeRecord,
    "mirror_trade": MirrorTradeRecord,
    "gold_exports": GoldExportRecord,
    "smuggling_incidents": SmugglingIncidentRecord,
}


class DatasetRegistry(BaseModel):
    name: str
    path: str
    type: str
    frequency: str
    primary_key: str
    description: str


def get_schema_for(dataset_name: str) -> Optional[Type[BaseModel]]:
    return SCHEMA_REGISTRY.get(dataset_name)


def infer_schema(df: pd.DataFrame, dataset_name: str) -> Dict:
    schema = get_schema_for(dataset_name)
    if schema is None:
        return {"fields": list(df.columns), "dtypes": {c: str(df[c].dtype) for c in df.columns}}
    return {
        "fields": list(schema.model_fields.keys()),
        "dtypes": {f: v.annotation.__name__ for f, v in schema.model_fields.items()},
    }
