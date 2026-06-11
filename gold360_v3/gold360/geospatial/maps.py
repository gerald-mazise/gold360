from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd

from gold360.utils.logging import setup_logging

logger = setup_logging(__name__)


class MapGenerator:
    @staticmethod
    def create_base_map(location: Tuple[float, float] = (-19.0, 29.5),
                         zoom_start: int = 7) -> folium.Map:
        m = folium.Map(
            location=location,
            zoom_start=zoom_start,
            tiles="CartoDB dark_matter",
            control_scale=True,
        )
        return m

    @staticmethod
    def add_mine_layer(m: folium.Map, df: pd.DataFrame,
                        lat_col: str = "mine_latitude",
                        lon_col: str = "mine_longitude",
                        risk_col: Optional[str] = None,
                        label_col: Optional[str] = None) -> folium.Map:
        for _, row in df.iterrows():
            lat, lon = row.get(lat_col), row.get(lon_col)
            if pd.isna(lat) or pd.isna(lon):
                continue

            risk = row.get(risk_col, 0) if risk_col else 0
            if pd.isna(risk):
                risk = 0

            if risk > 75:
                color = "#E53E3E"
            elif risk > 50:
                color = "#D69E2E"
            elif risk > 25:
                color = "#DD6B20"
            else:
                color = "#38A169"

            label = str(row.get(label_col, "")) if label_col else ""
            popup_text = f"<b>{label}</b><br>Risk Score: {risk:.1f}" if label else f"Risk: {risk:.1f}"

            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=popup_text,
            ).add_to(m)

        return m

    @staticmethod
    def add_heatmap_layer(m: folium.Map, df: pd.DataFrame,
                           lat_col: str = "mine_latitude",
                           lon_col: str = "mine_longitude",
                           weight_col: Optional[str] = None) -> folium.Map:
        from folium.plugins import HeatMap
        points = []
        for _, row in df.iterrows():
            lat, lon = row.get(lat_col), row.get(lon_col)
            if pd.isna(lat) or pd.isna(lon):
                continue
            weight = row.get(weight_col, 1) if weight_col and weight_col in row else 1
            if pd.isna(weight):
                weight = 1
            points.append([lat, lon, weight])
        HeatMap(points, radius=15, blur=10, max_zoom=1).add_to(m)
        return m

    @staticmethod
    def add_fgr_offices(m: folium.Map, df: pd.DataFrame) -> folium.Map:
        for _, row in df.iterrows():
            lat, lon = row.get("latitude"), row.get("longitude")
            if pd.isna(lat) or pd.isna(lon):
                continue
            folium.Marker(
                location=[lat, lon],
                popup=row.get("office_name", ""),
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
        return m
