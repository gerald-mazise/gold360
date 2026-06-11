import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 42
N_SAMPLES = 100


@pytest.fixture(scope="session")
def seed():
    return SEED


@pytest.fixture(scope="session")
def sample_mine_data():
    np.random.seed(SEED)
    return pd.DataFrame({
        "mine_id": [f"M{i:04d}" for i in range(N_SAMPLES)],
        "province": np.random.choice(
            ["Mashonaland West", "Mashonaland Central", "Midlands",
             "Manicaland", "Matabeleland South"], N_SAMPLES
        ),
        "month_year": pd.date_range("2020-01-01", periods=N_SAMPLES, freq="ME"),
        "delivery_volume_kg": np.random.exponential(50, N_SAMPLES),
        "ore_grade": np.random.uniform(0.5, 5.0, N_SAMPLES),
        "recovery_rate": np.random.uniform(0.6, 0.95, N_SAMPLES),
        "energy_availability": np.random.uniform(0.3, 1.0, N_SAMPLES),
        "rainfall_mm": np.random.exponential(80, N_SAMPLES),
        "fx_spread": np.random.uniform(0, 100, N_SAMPLES),
        "mine_latitude": np.random.uniform(-22.0, -16.0, N_SAMPLES),
        "mine_longitude": np.random.uniform(25.0, 33.0, N_SAMPLES),
        "is_synthetic": np.random.choice([0, 1], N_SAMPLES, p=[0.8, 0.2]),
    })


@pytest.fixture(scope="session")
def sample_fgr_data():
    return pd.DataFrame({
        "office_name": ["Harare", "Bulawayo", "Mutare", "Gweru", "Kadoma"],
        "latitude": [-17.8252, -20.1500, -18.9667, -19.4500, -18.3167],
        "longitude": [31.0335, 28.5833, 32.6667, 29.8167, 29.8833],
    })


@pytest.fixture(scope="session")
def sample_mirror_trade():
    np.random.seed(SEED)
    return pd.DataFrame({
        "year": [2020, 2021, 2022, 2023, 2024],
        "zim_exports_kg": [15, 18, 14, 20, 17],
        "partner_imports_kg": [28, 32, 30, 35, 33],
    })


@pytest.fixture(scope="session")
def sample_policy_events():
    np.random.seed(SEED)
    return pd.DataFrame({
        "event_date": pd.date_range("2020-01-01", periods=20, freq="QE"),
        "event_type": np.random.choice(
            ["regulatory", "fiscal", "enforcement", "trade"], 20
        ),
        "impact_score": np.random.uniform(-2, 2, 20),
    })
