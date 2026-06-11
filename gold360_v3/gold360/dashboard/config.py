from pathlib import Path
from typing import Dict, Optional

import streamlit as st

from gold360.utils.config import get_project_root


class DashboardConfig:
    def __init__(self):
        self.root = get_project_root()
        self.title = "GOLD360 — Intelligence Platform"
        self.subtitle = "Zimbabwe Gold Ecosystem — Economic Intelligence & Structural Risk Assessment"
        self.page_icon = ""
        self.layout = "wide"
        self.initial_sidebar_state = "expanded"
        self.menu_items: Dict[str, str] = {
            "Get Help": None,
            "Report a Bug": None,
            "About": "GOLD360 v1.0 — Intelligence Platform for Zimbabwe's Gold Ecosystem",
        }

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "raw"
