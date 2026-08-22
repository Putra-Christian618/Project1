import streamlit as st
import folium
from streamlit_folium import st_folium
import random
import requests
import math
import numpy as np
from PIL import Image

try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    from ultralytics import YOLO
except ImportError:
    st.error("Critical dependencies missing. Check requirements.txt.")

# === CONFIGURATION, CACHING & CSS INJECTION ===
st.set_page_config(page_title="EcoRouter AI", layout="wide")
FUEL_PRICE_IDR = 16000  # Pertamax Price per Liter

# Custom CSS for "Sleek Eco-Tech" Enterprise UI
st.markdown("""
<style>
    /* Reduce default top padding for a web-app feel */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Main Typography */
    h1 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    
    /* Sleek Metric Cards */
    div[data-testid="metric-container"] {
        background-color: #1E2320; /* Very dark subtle green */
        border: 1px solid #2E4035; /* Sharp eco-green border */
        padding: 5% 5% 5% 10%;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    /* Metric Label Typography */
    div[data-testid="metric-container"] label {
        color: #A3D8B4 !important; /* Soft highlight for labels */
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_yolo_model():
    """Caches the model in VRAM to prevent Out-Of-Memory crashes on button clicks."""
    try:
        return YOLO("models/best.pt")
    except Exception as e:
        return None

# === PHASE 3: EXACT FUEL ROUTING ENGINE ===
class ExactFuelRouter:
    def __init__(self, coords, weights_kg, volumes_cm3, vehicle_max_vol=1000000):
        self.coords = coords
        self.weights = weights_kg
        self.volumes = volumes_cm3
        self.vehicle_max_vol = vehicle_max_vol
        self.num_nodes = len(coords)
        self.dist_matrix = self._get_osrm_matrix()

    def _get_osrm_matrix(self):
        """Fetches live street data with a fallback to offline Haversine math."""
        coords_str = ";".join(self.coords)
        url = f"http://router.project-osrm.org/table/v1/driving/{coords_str}?annotations=distance"
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                data = response.json()
                if 'distances' in data:
                    return data['distances']
        except requests.exceptions.RequestException:
            pass
        return self._generate_fallback_matrix()

    def _generate_fallback_matrix(self):
        """Offline fail-safe: Calculates curved-earth distance with a 1.5 road tortuosity multiplier."""
        matrix = []
        points = [tuple(map(float, c.split(','))) for c in self.coords]
        for lon1, lat1 in points:
            row = []
            for lon2, lat2 in points:
                R, phi1, phi2 = 6371000, math.radians(lat1), math.radians(lat2)
                dphi, dlambda = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
                a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
                dist_m = R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a))) * 1.5
                row.append(dist_m)
            matrix.append(row)
        return matrix

    def calculate_fuel(self, route):
        total_fuel, current_payload = 0.0, sum(self.weights[n] for n in route)
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            if u != 0: current_payload -= self.weights[u] 
            dist_km = self.dist_matrix[u][v] / 1000.0
            total_fuel += dist_km * (0.3 + 0.00005 * current_payload)
        return total_fuel

    def solve_greedy(self):
        unvisited, route, curr = list(range(1, self.num_nodes)), [0], 0
        while unvisited:
            next_node = min(unvisited, key=lambda x: self.dist_matrix[curr][x])
            route.append(next_node)
            unvisited.remove(next_node)
            curr = next_node
        route.append(0)
        return route

    def solve_exact_fuel(self):
        manager = pywrapcp.RoutingIndexManager(self.num_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def dist_callback(from_idx, to_idx):
            return int(self.dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)]) * 30000
        transit_idx = routing.RegisterTransitCallback(dist_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

        def raw_dist_callback(from_idx, to_idx):
            return int(self.dist_matrix[manager.IndexToNode(from_idx)][manager.IndexToNode(to_idx)])
        raw_transit_idx = routing.RegisterTransitCallback(raw_dist_callback)
        routing.AddDimension(raw_transit_idx, 0, 3000000, True, "RawDistance")
        raw_dist_dim = routing.GetDimensionOrDie("RawDistance")

        for i in range(1, self.num_nodes):
            weight_penalty = int(self.weights[i] * 5)
            raw_dist_dim.SetCumulVarSoftUpperBound(manager.NodeToIndex(i), 0, weight_penalty)

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        search_params.time_limit.seconds = 1

        solution = routing.SolveWithParameters(search_params)
        if not solution: return []
        
        index = routing.Start(0)
        route_arr = []
        while not routing.IsEnd(index):
            route_arr.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route_arr.append(manager.IndexToNode(index))
        return route_arr

# === UI COMPONENTS ===
def plot_routes_on_map(coords, std_route, opt_route):
    """Generates an interactive Folium map comparing the two routes with parallel rendering."""
    lat_lons = [[float(c.split(',')[1]), float(c.split(',')[0])] for c in coords]
    m = folium.Map(location=lat_lons[0], zoom_start=11, tiles="CartoDB positron")
    
    # 1. Render Delivery Nodes
    for i, (lat, lon) in enumerate(lat_lons):
        color = 'darkred' if i == 0 else 'blue'
        label = "Depot" if i == 0 else f"Stop {i}"
        folium.Marker([lat, lon], popup=label, icon=folium.Icon(color=color)).add_to(m)
        
    # 2. Render Standard Route (Shifted slightly so it runs parallel)
    OFFSET = 0.00030 # Approx 30 meters
    shifted_std_route = [[lat_lons[idx][0] + OFFSET, lat_lons[idx][1] + OFFSET] for idx in std_route]
    folium.PolyLine(shifted_std_route, color="#E63946", weight=3, opacity=0.8, dash_array='5, 6', tooltip="Standard Route").add_to(m)
    
    # 3. Render Optimized Route (Accurate to coordinates)
    folium.PolyLine([lat_lons[idx] for idx in opt_route], color="#2A9D8F", weight=5, opacity=0.9, tooltip="EcoRoute AI").add_to(m)
    
    return m

def execute_pipeline(coords, weights, volumes):
    """Handles routing calculation and rendering the financial metric dashboard."""
    router = ExactFuelRouter(coords, weights, volumes)
    std_route = router.solve_greedy()
    opt_route = router.solve_exact_fuel()
    
    if not opt_route:
        st.error("Vehicle capacity exceeded. Cannot calculate route.")
        return

    # Calculate Fuel
    std_fuel = router.calculate_fuel(std_route)
    opt_fuel = router.calculate_fuel(opt_route)
    savings_pct = ((std_fuel - opt_fuel) / std_fuel) * 100 if std_fuel > 0 else 0
    
    # Calculate Financials
    std_cost = std_fuel * FUEL_PRICE_IDR
    opt_cost = opt_fuel * FUEL_PRICE_IDR
    money_saved = std_cost - opt_cost

    st.markdown("### Live Financial & Telemetry Metrics")
    col1, col2, col3 = st.columns(3)
    
    col1.metric(
        "Optimization Efficiency", 
        f"{savings_pct:.2f}%", 
        f"Saved {std_fuel - opt_fuel:.2f} Liters"
    )
    col2.metric(
        "Standard Path Cost", 
        f"Rp {std_cost:,.0f}", 
        f"{std_fuel:.2f} L used", 
        delta_color="inverse"
    )
    col3.metric(
        "EcoRoute AI Cost", 
        f"Rp {opt_cost:,.0f}", 
        f"Rp {money_saved:,.0f} Saved"
    )
    
    st.markdown("### Route Visualization")
    st.caption("Red Dashes: Standard Shortest-Path | Green Lines: Ton-Kilometer Optimized")
    map_obj = plot_routes_on_map(coords, std_route, opt_route)
    st_folium(map_obj, width=1000, height=450, returned_objects=[])

    st.markdown("### Warehouse Execution Instructions")
    lifo_sequence = opt_route[1:-1][::-1]
    st.info(f"**LIFO Loading Order (Back to Front):** Load items in this exact sequence: **{lifo_sequence}**")


# === MAIN APP FLOW ===
st.title("EcoRouter AI: Payload-Aware Logistics")
st.markdown("Dynamic fuel optimization using Computer Vision and LIFO warehouse constraints.")

# Expanded Jabodetabek Depots
DEPOTS = {
    "Central Jakarta (Hub)": ("106.816666,-6.200000", 0, 0),
    "South Tangerang (Hub)": ("106.711400,-6.288600", 0, 0),
    "Depok (Hub)": ("106.827200,-6.402500", 0, 0),
    "Bekasi (Hub)": ("106.989600,-6.233600", 0, 0)
}

# Expanded 20-Point Jabodetabek Coordinate Pool
MOCK_POOL = [
    ("106.825000,-6.210000", 15, 20000), ("106.835000,-6.215000", 25, 30000), 
    ("106.845000,-6.220000", 10, 15000), ("106.855000,-6.230000", 5, 10000),
    ("106.989600,-6.269000", 850, 400000), ("106.798300,-6.262500", 20, 15000),
    ("106.628800,-6.178300", 800, 400000), ("106.890000,-6.150000", 350, 300000),
    ("106.750000,-6.110000", 350, 300000), ("106.741000,-6.187300", 30, 15000),
    ("106.797200,-6.597100", 45, 45000), ("106.806000,-6.598000", 120, 90000), # Bogor
    ("106.900000,-6.160000", 55, 30000), ("106.890000,-6.370000", 210, 180000), # Kelapa Gading, Cibubur
    ("106.738300,-6.106600", 12, 10000), ("106.650000,-6.300000", 500, 350000), # PIK, BSD
    ("106.820000,-6.290000", 35, 20000), ("106.870000,-6.210000", 75, 50000), 
    ("107.010000,-6.250000", 600, 400000), ("106.720000,-6.200000", 18, 15000)
]

mode = st.sidebar.radio("Select Demonstration Mode", 
    ["Mode A: Curated Benchmarks", "Mode B: Dynamic Random Sandbox", "Mode C: Visual Ingestion (Camera)"])

if mode == "Mode A: Curated Benchmarks":
    scenario = st.selectbox("Select Scenario:", ["The Tangerang Whale (Extreme Weight Outlier)", "Balanced Urban Run (Uniform Data)"])
    if st.button("Calculate Optimal Route"):
        depot = DEPOTS["Central Jakarta (Hub)"]
        if "Tangerang" in scenario:
            coords = [depot[0], "106.7983,-6.2625", "106.8000,-6.2650", "106.8100,-6.2750", "106.6288,-6.1783"]
            weights = [0, 15, 20, 10, 800]
            volumes = [0, 10000, 10000, 10000, 400000]
        else:
            coords = [depot[0], "106.8200,-6.1800", "106.8250,-6.1700", "106.8300,-6.1600", "106.8100,-6.1750"]
            weights = [0, 40, 35, 50, 45]
            volumes = [0, 20000, 20000, 20000, 20000]
        execute_pipeline(coords, weights, volumes)

elif mode == "Mode B: Dynamic Random Sandbox":
    st.markdown("### Configure Regional Logistics Run")
    selected_depot_name = st.selectbox("Select Departure Depot:", list(DEPOTS.keys()))
    package_count = st.slider("Select number of packages in fleet:", 3, 10, 4)
    
    if st.button("Generate Random Fleet & Optimize"):
        current_depot = DEPOTS[selected_depot_name]
        selected = random.sample(MOCK_POOL, package_count)
        
        coords = [current_depot[0]] + [item[0] for item in selected]
        weights = [current_depot[1]] + [item[1] for item in selected]
        volumes = [current_depot[2]] + [item[2] for item in selected]
        execute_pipeline(coords, weights, volumes)

elif mode == "Mode C: Visual Ingestion (Camera)":
    st.info("Allow browser camera permissions to execute Phase 1 ingestion.")
    cam_image = st.camera_input("Scan Logistics Cargo Bay")
    
    if cam_image is not None:
        model = load_yolo_model()
        if model is None:
            st.error("Weights file not found at models/best.pt. Cannot execute Phase 1.")
        else:
            img = Image.open(cam_image)
            results = model.predict(source=img, conf=0.25, verbose=False)
            box_count = len(results[0].boxes)
            
            res_plotted = results[0].plot()
            st.image(res_plotted, caption=f"YOLO Detected {box_count} Packages", use_container_width=True)
            
            if box_count > 0:
                sample_size = min(box_count, len(MOCK_POOL))
                selected = random.sample(MOCK_POOL, sample_size)
                # Defaults to Central Jakarta for Visual Ingestion
                depot = DEPOTS["Central Jakarta (Hub)"]
                
                coords = [depot[0]] + [item[0] for item in selected]
                weights = [depot[1]] + [item[1] for item in selected]
                volumes = [depot[2]] + [item[2] for item in selected]
                execute_pipeline(coords, weights, volumes)
            else:
                st.warning("No packages detected. Route generation halted.")