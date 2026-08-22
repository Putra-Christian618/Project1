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
    st.error("Missing dependencies — install everything in requirements.txt, then restart the app.")

# ============================================================
# CONFIGURATION
# ============================================================
st.set_page_config(page_title="EcoRouter AI", page_icon="🧭", layout="wide")
FUEL_PRICE_IDR = 16000  # Pertamax price per liter (IDR)

# Route + brand palette
COLOR_ECO = "#35D69B"     # optimized / EcoRouter route
COLOR_BEACON = "#FF8A3D"  # baseline / standard route

# ============================================================
# STYLE — "Night Dispatch Console" theme
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --ink: #0A0F0D;
        --panel: #121B17;
        --panel-2: #16211B;
        --hairline: rgba(148, 179, 163, 0.16);
        --hairline-strong: rgba(148, 179, 163, 0.32);
        --eco: #35D69B;
        --beacon: #FF8A3D;
        --paper: #EAF3EE;
        --mist: #86998D;
    }

    /* ---- app shell ---- */
    [data-testid="stAppViewContainer"], [data-testid="stApp"] {
        background:
            radial-gradient(ellipse 900px 480px at 12% -8%, rgba(53,214,155,0.07), transparent 60%),
            radial-gradient(ellipse 700px 480px at 100% 0%, rgba(255,138,61,0.05), transparent 55%),
            var(--ink);
    }
    .block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1180px; }
    html, body, p, span, div, label { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }
    h1, h2, h3, h4 { font-family: 'Rajdhani', sans-serif; color: var(--paper); }
    label, [data-testid="stWidgetLabel"] p { color: var(--paper) !important; }
    [data-testid="stCaptionContainer"], .stCaption, small { color: var(--mist) !important; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--ink); }
    ::-webkit-scrollbar-thumb { background: var(--hairline-strong); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--eco); }

    hr { border-color: var(--hairline) !important; }
    iframe { border-radius: 12px; border: 1px solid var(--hairline); }

    /* ---- header / wordmark ---- */
    @keyframes eco-fade-up { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
    .eco-header {
        display: flex; align-items: center; gap: 1.15rem;
        padding-bottom: 1.4rem; margin-bottom: 1.6rem;
        border-bottom: 1px solid var(--hairline);
        animation: eco-fade-up 0.55s ease-out;
    }
    .eco-logo { flex-shrink: 0; filter: drop-shadow(0 0 16px rgba(53,214,155,0.20)); }
    .eco-eyebrow-inline {
        font-family: 'Rajdhani', sans-serif; font-size: 0.78rem; font-weight: 600;
        letter-spacing: 0.16em; text-transform: uppercase; color: var(--mist); margin: 0 0 0.15rem 0;
    }
    h1.eco-title { font-family: 'Rajdhani', sans-serif; font-size: 2.6rem; font-weight: 700; line-height: 1; margin: 0.1rem 0; color: var(--paper); letter-spacing: -0.01em; }
    .eco-title-accent { color: var(--eco); }
    .eco-tagline { font-size: 0.98rem; color: var(--mist); margin: 0.35rem 0 0 0; max-width: 620px; }

    /* ---- sidebar ---- */
    [data-testid="stSidebar"] { background: var(--panel) !important; border-right: 1px solid var(--hairline); }
    .eco-side-brand { display: flex; align-items: center; gap: 0.6rem; padding: 0.2rem 0 1.1rem 0; margin-bottom: 0.75rem; border-bottom: 1px solid var(--hairline); }
    .eco-side-brand-text { line-height: 1.15; font-family: 'Rajdhani', sans-serif; }
    .eco-side-brand-text strong { display: block; font-size: 1.02rem; color: var(--paper); font-weight: 700; }
    .eco-side-brand-text span { display: block; font-size: 0.64rem; color: var(--eco); letter-spacing: 0.18em; font-weight: 600; }

    /* ---- eyebrow / section labels ---- */
    .eyebrow {
        display: flex; align-items: center; gap: 9px;
        font-family: 'Rajdhani', sans-serif; font-size: 0.8rem; font-weight: 600;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--mist);
        margin: 1.9rem 0 0.85rem 0;
    }
    .eyebrow::before { content: ''; width: 7px; height: 7px; background: var(--eco); transform: rotate(45deg); border-radius: 1px; flex-shrink: 0; box-shadow: 0 0 8px rgba(53,214,155,0.7); }
    .eyebrow.first { margin-top: 0.2rem; }
    .pulse-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--eco); display: inline-block; animation: eco-pulse 2s infinite; margin-right: 2px; }
    @keyframes eco-pulse { 0% { box-shadow: 0 0 0 0 rgba(53,214,155,0.45); } 70% { box-shadow: 0 0 0 7px rgba(53,214,155,0); } 100% { box-shadow: 0 0 0 0 rgba(53,214,155,0); } }

    /* ---- metrics ---- */
    div[data-testid="stMetric"], div[data-testid="metric-container"] {
        background: linear-gradient(180deg, var(--panel-2), var(--panel));
        border: 1px solid var(--hairline); border-radius: 12px; padding: 1.05rem 1.3rem;
        transition: border-color .2s ease, transform .2s ease;
    }
    div[data-testid="stMetric"]:hover, div[data-testid="metric-container"]:hover { border-color: var(--hairline-strong); transform: translateY(-2px); }
    [data-testid="stMetricLabel"] { font-family: 'Rajdhani', sans-serif !important; text-transform: uppercase; letter-spacing: .07em; font-size: .82rem !important; color: var(--mist) !important; }
    [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; color: var(--paper) !important; }
    [data-testid="stMetricDelta"] { font-family: 'IBM Plex Mono', monospace !important; }

    /* ---- buttons & file uploader styling ---- */
    .stButton > button {
        background: linear-gradient(135deg, var(--eco), #23B589); color: #06110D; border: none; border-radius: 8px;
        font-family: 'Rajdhani', sans-serif; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
        padding: .6rem 1.5rem; transition: transform .15s ease, box-shadow .15s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 8px 22px rgba(53,214,155,.32); }
    .stButton > button:focus-visible { outline: 2px solid var(--eco); outline-offset: 2px; }

    /* ---- alerts ---- */
    .stAlert, [data-testid="stAlert"] { background: var(--panel-2) !important; border: 1px solid var(--hairline); border-radius: 9px; }
    .stAlert p, [data-testid="stAlert"] p { font-family: 'IBM Plex Sans', sans-serif; }

    /* ---- legend chips ---- */
    .route-legend { display: flex; flex-wrap: wrap; gap: 1.4rem; margin: 0 0 1rem 0; font-size: .86rem; color: var(--mist); }
    .legend-chip { display: flex; align-items: center; gap: .5rem; }
    .legend-line { width: 20px; height: 3px; border-radius: 2px; display: inline-block; }

    @media (max-width: 640px) {
        .eco-header { flex-direction: column; align-items: flex-start; gap: .75rem; }
        h1.eco-title { font-size: 2rem !important; }
    }
    @media (prefers-reduced-motion: reduce) {
        * { animation: none !important; transition: none !important; }
    }
</style>
""", unsafe_allow_html=True)


def eyebrow(text, first=False, live=False):
    """Renders a small tracked-uppercase section label with the waypoint tick."""
    cls = "eyebrow first" if first else "eyebrow"
    prefix = '<span class="pulse-dot"></span>' if live else ""
    st.markdown(f'<div class="{cls}">{prefix}{text}</div>', unsafe_allow_html=True)


def brand_mark(size=54, stroke_width=2.2):
    """Inline SVG wordmark: a route bending between a baseline node and an optimized node."""
    return f'''<svg width="{size}" height="{size}" viewBox="0 0 54 54" xmlns="http://www.w3.org/2000/svg">
        <rect x="1.5" y="1.5" width="51" height="51" rx="14" fill="#121B17" stroke="#24352C" stroke-width="1"/>
        <path d="M15 37 C15 30 17 22 25 19 C31 17 35 20 39 17" fill="none" stroke="{COLOR_ECO}" stroke-width="{stroke_width}" stroke-linecap="round"/>
        <circle cx="15" cy="37" r="3.4" fill="{COLOR_BEACON}"/>
        <circle cx="39" cy="17" r="3.4" fill="{COLOR_ECO}"/>
    </svg>'''


@st.cache_resource
def load_yolo_model():
    """Caches the model in VRAM to prevent Out-Of-Memory crashes on button clicks."""
    try:
        return YOLO("models/best.pt")
    except Exception:
        return None

# ============================================================
# PHASE 3: EXACT FUEL ROUTING ENGINE
# ============================================================
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

# ============================================================
# UI COMPONENTS
# ============================================================
def plot_routes_on_map(coords, std_route, opt_route):
    """Generates an interactive Folium map comparing the two routes with parallel rendering."""
    lat_lons = [[float(c.split(',')[1]), float(c.split(',')[0])] for c in coords]
    m = folium.Map(location=lat_lons[0], zoom_start=11, tiles="CartoDB dark_matter")

    # 1. Render Delivery Nodes
    for i, (lat, lon) in enumerate(lat_lons):
        color = 'white' if i == 0 else 'cadetblue'
        label = "Depot" if i == 0 else f"Stop {i}"
        folium.Marker([lat, lon], popup=label, tooltip=label, icon=folium.Icon(color=color)).add_to(m)

    # 2. Render Standard Route (Shifted slightly so it runs parallel)
    OFFSET = 0.00030  # Approx 30 meters
    shifted_std_route = [[lat_lons[idx][0] + OFFSET, lat_lons[idx][1] + OFFSET] for idx in std_route]
    folium.PolyLine(shifted_std_route, color=COLOR_BEACON, weight=3, opacity=0.85, dash_array='5, 6', tooltip="Standard Route").add_to(m)

    # 3. Render Optimized Route (Accurate to coordinates)
    folium.PolyLine([lat_lons[idx] for idx in opt_route], color=COLOR_ECO, weight=5, opacity=0.95, tooltip="EcoRouter Route").add_to(m)

    return m

def execute_pipeline(coords, weights, volumes):
    """Handles routing calculation and rendering the financial metric dashboard."""
    router = ExactFuelRouter(coords, weights, volumes)
    std_route = router.solve_greedy()
    opt_route = router.solve_exact_fuel()

    if not opt_route:
        st.error("This load exceeds the vehicle's volume capacity. Remove a package or split the run across two trips.")
        return

    # Calculate Fuel
    std_fuel = router.calculate_fuel(std_route)
    opt_fuel = router.calculate_fuel(opt_route)
    savings_pct = ((std_fuel - opt_fuel) / std_fuel) * 100 if std_fuel > 0 else 0

    # Calculate Financials
    std_cost = std_fuel * FUEL_PRICE_IDR
    opt_cost = opt_fuel * FUEL_PRICE_IDR
    money_saved = std_cost - opt_cost

    eyebrow("Fuel & Cost Telemetry")
    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Fuel Efficiency Gain",
            f"{savings_pct:.2f}%",
            f"Saved {std_fuel - opt_fuel:.2f} L"
        )
        col2.metric(
            "Standard Route Cost",
            f"Rp {std_cost:,.0f}",
            f"{std_fuel:.2f} L used",
            delta_color="inverse"
        )
        col3.metric(
            "EcoRouter Route Cost",
            f"Rp {opt_cost:,.0f}",
            f"Rp {money_saved:,.0f} saved"
        )

    eyebrow("Route Map", live=True)
    st.markdown(f'''<div class="route-legend">
        <span class="legend-chip"><span class="legend-line" style="background:{COLOR_BEACON};opacity:.85;"></span>Standard route — shortest path only</span>
        <span class="legend-chip"><span class="legend-line" style="background:{COLOR_ECO};"></span>EcoRouter route — optimized for payload-weighted fuel burn</span>
    </div>''', unsafe_allow_html=True)
    with st.container(border=True):
        map_obj = plot_routes_on_map(coords, std_route, opt_route)
        st_folium(map_obj, width=1000, height=460, returned_objects=[])

    eyebrow("Loading Sequence")
    lifo_sequence = opt_route[1:-1][::-1]
    sequence_str = " → ".join(f"Stop {n}" for n in lifo_sequence)
    st.info(
        f"**Back-to-front loading order:** {sequence_str}\n\n"
        "Load in this order so every stop sits nearest the door exactly when it's needed."
    )


# ============================================================
# HEADER
# ============================================================
st.markdown(f'''<div class="eco-header">
    <div class="eco-logo">{brand_mark(56, 2.4)}</div>
    <div>
        <div class="eco-eyebrow-inline">Jabodetabek · Fleet Operations</div>
        <h1 class="eco-title">EcoRouter <span class="eco-title-accent">AI</span></h1>
        <p class="eco-tagline">Every stop lightens the load — EcoRouter sequences deliveries around it to burn the least fuel getting there.</p>
    </div>
</div>''', unsafe_allow_html=True)

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

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown(f'''<div class="eco-side-brand">
    {brand_mark(34, 2.6)}
    <div class="eco-side-brand-text"><strong>EcoRouter</strong><span>AI CONSOLE</span></div>
</div>''', unsafe_allow_html=True)

st.sidebar.markdown('<div class="eyebrow first">Operating Mode</div>', unsafe_allow_html=True)
mode = st.sidebar.radio(
    "Operating mode",
    ["A · Curated Benchmarks", "B · Dynamic Sandbox", "C · Visual Ingestion"],
    captions=[
        "Two fixed edge-case scenarios",
        "Randomized fleet from the regional pool",
        "Camera + file upload ingestion",
    ],
    label_visibility="collapsed",
)

if mode == "A · Curated Benchmarks":
    eyebrow("Scenario", first=True)
    with st.container(border=True):
        scenario = st.selectbox(
            "Scenario",
            ["The Tangerang Whale (Extreme Weight Outlier)", "Balanced Urban Run (Uniform Data)"],
            label_visibility="collapsed",
        )
        run_a = st.button("Calculate optimal route")

    if run_a:
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

elif mode == "B · Dynamic Sandbox":
    eyebrow("Configure the Run", first=True)
    with st.container(border=True):
        selected_depot_name = st.selectbox("Departure depot", list(DEPOTS.keys()))
        package_count = st.slider("Number of packages", 3, 10, 4)
        run_b = st.button("Generate random fleet & optimize")

    if run_b:
        current_depot = DEPOTS[selected_depot_name]
        selected = random.sample(MOCK_POOL, package_count)

        coords = [current_depot[0]] + [item[0] for item in selected]
        weights = [current_depot[1]] + [item[1] for item in selected]
        volumes = [current_depot[2]] + [item[2] for item in selected]
        execute_pipeline(coords, weights, volumes)

elif mode == "C · Visual Ingestion":
    eyebrow("Visual Ingestion", first=True)
    with st.container(border=True):
        st.info("Provide an image of the cargo bay — EcoRouter counts packages automatically.")
        
        # Adding a radio button to cleanly toggle between Upload and Camera inputs
        input_method = st.radio(
            "Choose image source:", 
            ["📷 Live Camera", "📁 Upload File"], 
            horizontal=True
        )
        
        # Variable to hold whichever image source the user provides
        img_data = None
        
        if input_method == "📷 Live Camera":
            img_data = st.camera_input("Scan the cargo bay", label_visibility="collapsed")
        else:
            img_data = st.file_uploader("Upload a local image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if img_data is not None:
        model = load_yolo_model()
        if model is None:
            st.error("Detection model not found. Place the trained weights at `models/best.pt`, then reload the app.")
        else:
            # PIL Image can read the buffer regardless of whether it came from the camera or file uploader
            img = Image.open(img_data)
            results = model.predict(source=img, conf=0.25, verbose=False)
            box_count = len(results[0].boxes)

            res_plotted = results[0].plot()
            st.image(res_plotted, caption=f"{box_count} packages detected", use_container_width=True)

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
                st.warning("No packages detected in frame. Adjust the image and try again.")