# GOLD360 — Geospatial Outputs

## Modules

| Module | Class | Purpose |
|--------|-------|---------|
| `distances.py` | `GeoDistances` | Haversine distance, distance matrix, nearest FGR/border |
| `corridors.py` | `CorridorAnalyzer` | Corridor risk scores weighted by border post |
| `clusters.py` | `SpatialCluster` | DBSCAN spatial clustering |
| `maps.py` | `MapGenerator` | Folium dark-themed risk maps |

## Border Posts (8)

| Post | Weight | Coordinates |
|------|--------|-------------|
| Beitbridge | 0.30 | -22.2333, 29.9833 |
| Plumtree | 0.25 | -20.4833, 27.8167 |
| Mutare | 0.20 | -18.9667, 32.6167 |
| Chirundu | 0.15 | -16.0333, 28.8500 |
| Victoria Falls | 0.10 | -17.9333, 25.8500 |
| Nyamapanda | 0.10 | -16.4667, 32.7833 |
| Forbes | 0.10 | -18.5833, 32.7500 |
| Kazungula | 0.10 | -17.7833, 25.2667 |

## Provinces (10)

| Province | Approximate Center |
|----------|-------------------|
| Bulawayo | -20.15, 28.58 |
| Harare | -17.83, 31.05 |
| Manicaland | -18.96, 32.62 |
| Mashonaland Central | -17.15, 31.20 |
| Mashonaland East | -18.58, 31.95 |
| Mashonaland West | -17.83, 29.50 |
| Masvingo | -20.07, 30.85 |
| Matabeleland North | -18.50, 27.50 |
| Matabeleland South | -21.50, 29.50 |
| Midlands | -19.50, 30.00 |

## Corridor Risk Weights

Corridor risk is computed as a weighted sum of border post proximity:

```python
corridor_risk = (
    weight Beitbridge * proximity_Beitbridge +
    weight Plumtree * proximity_Plumtree +
    weight Mutare * proximity_Mutare +
    weight Chirundu * proximity_Chirundu +
    weight Victoria_Falls * proximity_VF +
    ...
)
```

**Rationale:** Beitbridge carries highest weight (30%) due to highest traffic volume.

## FGR Buying Offices

Fidelity Gold Refinery (FGR) buying offices are the official delivery points for gold. Distance to FGR affects delivery friction.

**Feature:** `fgr_access = 1 / (1 + distance_to_fidelity_km / 200)`

## DBSCAN Clustering

| Parameter | Value |
|-----------|-------|
| Algorithm | DBSCAN |
| eps | 0.5 degrees (~55 km) |
| min_samples | 5 |
| Scaler | StandardScaler |
| Feature | [latitude, longitude] |

Clusters identify spatial groupings of mines. Cluster membership can be used for regional risk analysis.

## Map Layers

| Layer | Type | Description |
|-------|------|-------------|
| Mine markers | CircleMarker | Color-coded by risk tier (green/amber/orange/red) |
| Province choropleth | Choropleth | Province-level risk aggregation |
| Heatmap | HeatMap | Risk density surface |
| FGR offices | Marker | Gold diamond icons |
| Border posts | Marker | Red triangle icons |
| Corridors | PolyLine | Border-to-mine corridors |
| Hotspots | CircleMarker | Detected anomaly clusters |

## Map Configuration

| Setting | Value |
|---------|-------|
| Base tile | `CartoDB dark_matter` |
| Default zoom | 7 (Zimbabwe national) |
| Marker radius | 8px |
| Heatmap radius | 25 |
| Heatmap blur | 15 |

## Geospatial Figures

| Figure | Description |
|--------|-------------|
| `FIG_67_province_risk_map` | Province-level risk choropleth |
| `FIG_68_mine_risk_heatmap` | Mine-level risk heatmap |
| `FIG_69_fgr_office_coverage` | FGR office coverage areas |
| `FIG_70_border_risk_map` | Border proximity risk |
| `FIG_71_corridor_risk_map` | Corridor risk analysis |
| `FIG_72_spatial_cluster_map` | DBSCAN spatial clusters |
| `FIG_73_risk_density_surface` | Risk density surface |
| `FIG_74_hotspot_analysis` | Hotspot analysis |

## Code Example

```python
from gold360.geospatial.maps import MapGenerator
from gold360.geospatial.clusters import SpatialCluster

# Generate risk map
gen = MapGenerator()
m = gen.create_base_map(zoom=7)
m = gen.add_mine_markers(m, mine_data, risk_scores)
m = gen.add_heatmap(m, mine_data, risk_scores)

# Cluster analysis
cluster = SpatialCluster(eps=0.5, min_samples=5)
labels = cluster.fit(mine_coords)
```
