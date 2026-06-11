import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium

from gold360.dashboard.theme import Gold360Theme

Gold360Theme.apply_custom_css()

st.markdown("<h1 style='color:#D4AF37;font-size:1.8rem;margin:0;'>Geospatial Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94A3B8;margin:0.3rem 0 1rem 0;'>Spatial clustering, corridor risk, and interactive risk maps</p>", unsafe_allow_html=True)

# ======================================================================
# Zimbabwe Mine Locations (Based on real mining provinces)
# ======================================================================
np.random.seed(42)

# Real Zimbabwe mining provinces with approximate centers
province_coords = {
    "Mashonaland West": (-19.5, 29.5, 25),
    "Mashonaland Central": (-17.5, 31.0, 20),
    "Midlands": (-19.5, 30.0, 18),
    "Manicaland": (-19.0, 32.5, 12),
    "Matabeleland South": (-21.0, 30.0, 10),
}

mines = []
for prov, (lat, lon, count) in province_coords.items():
    for i in range(count):
        mines.append({
            "mine_id": f"M{len(mines)+1:04d}",
            "lat": lat + np.random.normal(0, 0.5),
            "lon": lon + np.random.normal(0, 0.8),
            "province": prov,
            "risk_score": np.random.beta(2, 5) * 100,
        })

mines_df = pd.DataFrame(mines)

# Real Zimbabwe border crossings
border_crossings = [
    {"name": "Beitbridge", "lat": -22.22, "lon": 29.99, "country": "South Africa", "traffic": "High"},
    {"name": "Plumtree", "lat": -20.48, "lon": 27.82, "country": "Botswana", "traffic": "Medium"},
    {"name": "Mutare", "lat": -18.97, "lon": 32.67, "country": "Mozambique", "traffic": "Medium"},
    {"name": "Chirundu", "lat": -16.03, "lon": 28.85, "country": "Zambia", "traffic": "Low"},
    {"name": "Victoria Falls", "lat": -17.93, "lon": 25.85, "country": "Zambia", "traffic": "Low"},
]

# Fidelity Gold Refinery (Harare)
fgr = {"name": "FGR Harare", "lat": -17.83, "lon": 31.05}

# Major airports
airports = [
    {"name": "Harare International", "lat": -17.93, "lon": 31.09},
    {"name": "Joshua Mqabuko Nkomo", "lat": -20.13, "lon": 28.86},
    {"name": "Victoria Falls", "lat": -18.09, "lon": 25.84},
]

# ======================================================================
# Interactive Risk Map
# ======================================================================
st.markdown("### Risk Map — Mine Locations & Border Corridors")
m = folium.Map(location=[-19.0, 30.0], zoom_start=7, tiles="CartoDB dark_matter")

# Add mines
for _, row in mines_df.iterrows():
    risk = row["risk_score"]
    if risk > 75:
        color = "#E53E3E"
    elif risk > 50:
        color = "#DD6B20"
    elif risk > 25:
        color = "#D69E2E"
    else:
        color = "#38A169"
    folium.CircleMarker(
        location=[row["lat"], row["lon"]], radius=4,
        color=color, fill=True, fill_opacity=0.7,
        popup=f"{row['mine_id']}<br>Province: {row['province']}<br>Risk: {row['risk_score']:.1f}",
    ).add_to(m)

# Add border crossings
for bc in border_crossings:
    folium.Marker(
        location=[bc["lat"], bc["lon"]],
        icon=folium.Icon(color="red", icon="crosshairs", prefix="fa"),
        popup=f"{bc['name']}<br>{bc['country']}<br>Traffic: {bc['traffic']}",
    ).add_to(m)
    folium.PolyLine(
        locations=[[bc["lat"], bc["lon"]], [fgr["lat"], fgr["lon"]]],
        color="#D4AF37", weight=2, opacity=0.6, dash_array="5",
    ).add_to(m)

# Add FGR
folium.Marker(
    location=[fgr["lat"], fgr["lon"]],
    icon=folium.Icon(color="gold", icon="university", prefix="fa"),
    popup="Fidelity Gold Refinery — Harare",
).add_to(m)

# Add airports
for ap in airports:
    folium.Marker(
        location=[ap["lat"], ap["lon"]],
        icon=folium.Icon(color="blue", icon="plane", prefix="fa"),
        popup=f"{ap['name']} Airport",
    ).add_to(m)

st_folium(m, width=1200, height=500)

# ======================================================================
# Corridor Risk + Province Distribution
# ======================================================================
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("### Corridor Risk Index")
    corridors = ["Beitbridge", "Plumtree", "Mutare", "Chirundu", "Victoria Falls"]
    corridor_weights = [0.35, 0.25, 0.20, 0.12, 0.08]
    corridor_mines = [28, 22, 18, 12, 5]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=corridors, y=corridor_weights, name="Risk Weight",
                         marker_color="#E53E3E", text=[f"{w:.0%}" for w in corridor_weights],
                         textposition="outside", textfont=dict(color="#E53E3E")))
    fig.add_trace(go.Bar(x=corridors, y=[m/sum(corridor_mines) for m in corridor_mines], name="Mine Share",
                         marker_color="#D4AF37", text=[f"{m}" for m in corridor_mines],
                         textposition="inside", textfont=dict(color="white")))
    fig.update_layout(barmode="group",
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748"),
        yaxis=dict(gridcolor="#2D3748", title="Share"),
        legend=dict(font=dict(color="#94A3B8")),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("### Risk by Province")
    province_risk = mines_df.groupby("province")["risk_score"].agg(["mean", "count"]).sort_values("mean", ascending=True)
    colors = ["#E53E3E" if v > 50 else "#DD6B20" if v > 35 else "#38A169" for v in province_risk["mean"].values]

    fig = go.Figure(go.Bar(x=province_risk["mean"], y=province_risk.index, orientation="h",
                           marker_color=colors,
                           text=[f"{v:.1f} ({c} mines)" for v, c in zip(province_risk["mean"], province_risk["count"])],
                           textposition="outside"))
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Avg Risk Score"),
        yaxis=dict(gridcolor="#2D3748"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Distance to Border + FGR Access
# ======================================================================
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### Distance to Border Distribution")
    # Calculate real distances to nearest border crossing
    from math import radians, sin, cos, sqrt, atan2
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    
    mines_df["border_dist"] = mines_df.apply(
        lambda row: min(haversine(row["lat"], row["lon"], bc["lat"], bc["lon"]) 
                       for bc in border_crossings), axis=1
    )
    
    fig = px.histogram(mines_df, x="border_dist", nbins=20,
                       color_discrete_sequence=["#D4AF37"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Distance to Nearest Border (km)"),
        yaxis=dict(gridcolor="#2D3748", title="Count"),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("### Risk vs Border Proximity")
    fig = px.scatter(mines_df, x="border_dist", y="risk_score",
                     color="province", opacity=0.7,
                     color_discrete_sequence=["#D4AF37", "#38A169", "#DD6B20", "#805AD5", "#E53E3E"])
    fig.update_layout(
        plot_bgcolor="#0F172A", paper_bgcolor="#0F172A",
        font_color="#94A3B8", xaxis=dict(gridcolor="#2D3748", title="Distance to Border (km)"),
        yaxis=dict(gridcolor="#2D3748", title="Risk Score"),
        legend=dict(font=dict(color="#94A3B8", size=9)),
        margin=dict(t=10, b=10, l=10, r=10), height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================
# Spatial Cluster Summary
# ======================================================================
st.markdown("### DBSCAN Cluster Summary")
cluster_data = pd.DataFrame({
    "Cluster": ["0 (Core)", "1 (Periphery)", "2 (Border)", "-1 (Noise)"],
    "Mines": [32, 25, 18, 10],
    "Avg Risk": [28.5, 42.1, 61.3, 73.8],
    "Color": ["#38A169", "#D69E2E", "#DD6B20", "#E53E3E"],
})
cols = st.columns(4)
for col, (_, row) in zip(cols, cluster_data.iterrows()):
    with col:
        st.markdown(f"""
        <div style='background:#1E293B;border-top:3px solid {row["Color"]};
            border-radius:0 0 8px 8px;padding:1rem;text-align:center;'>
            <div style='font-size:1.5rem;font-weight:700;color:{row["Color"]};'>{row["Mines"]}</div>
            <div style='font-size:0.8rem;color:#94A3B8;'>Mines — {row["Cluster"]}</div>
            <div style='font-size:0.75rem;color:#CBD5E1;margin-top:0.3rem;'>Avg Risk: {row["Avg Risk"]:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='color:#64748B;font-size:0.8rem;text-align:center;'>Mine locations are synthetic for development purposes. Border crossings and FGR locations are based on real Zimbabwe geography.</p>", unsafe_allow_html=True)
