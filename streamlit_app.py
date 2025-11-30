
import os
import math
import json
import heapq
import time
import logging
from typing import List, Tuple, Dict, Optional

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree

import streamlit as st
import folium
from folium.plugins import AntPath
from streamlit_folium import st_folium
import altair as alt

# matplotlib for colored segment plot (dark theme)
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# optional fuzzy matching
try:
    from rapidfuzz import process as fuzzy_process
    RAPIDFUZZ_AVAILABLE = True
except Exception:
    RAPIDFUZZ_AVAILABLE = False

# logging
LOGFILE = "streamlit_app.log"
logging.basicConfig(filename=LOGFILE, level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logging.info("App start")

# config
DATA_PATH = "./worldcitiespop.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
EARTH_R = 6371.0  # km

# session state defaults
if "tree_built" not in st.session_state:
    st.session_state["tree_built"] = False
if "tree" not in st.session_state:
    st.session_state["tree"] = None
if "dist_rad" not in st.session_state:
    st.session_state["dist_rad"] = None
if "indices" not in st.session_state:
    st.session_state["indices"] = None
if "edges" not in st.session_state:
    st.session_state["edges"] = None
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_route_df" not in st.session_state:
    st.session_state["last_route_df"] = None

# helper: check for local files that may shadow packages
def check_for_shadowing():
    bad_names = ["streamlit.py", "folium.py", "pandas.py", "numpy.py"]
    cwd_files = [f for f in os.listdir(".") if os.path.isfile(f)]
    conflicts = [f for f in bad_names if f in cwd_files]
    return conflicts

# ---------------------------
# Data & model helpers
# ---------------------------

@st.cache_data(show_spinner=False)
def load_and_prepare(path: str = DATA_PATH, dedupe: bool = True) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Put your CSV file there or change DATA_PATH.")
    df = pd.read_csv(path)

    # normalize columns (adjust to your CSV)
    df = df.rename(columns={
        "Country": "country",
        "City": "city",
        "AccentCity": "accent_city",
        "Region": "region",
        "Population": "population",
        "Latitude": "lat",
        "Longitude": "lon"
    })

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["population"] = pd.to_numeric(df.get("population", np.nan), errors="coerce")

    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    if dedupe:
        df = df.sort_values("population", ascending=False).groupby(["country", "city"], as_index=False).first().reset_index(drop=True)

    df["population_known"] = ~df["population"].isna()
    df["lon_mod"] = (df["lon"].astype(float) + 360.0) % 360.0
    df["lat_rad"] = np.deg2rad(df["lat"].astype(float))
    df["lon_rad"] = np.deg2rad(df["lon"].astype(float))
    df["country_code"] = df["country"].astype(str).str.lower()
    return df

def impute_population_by_neighbor_median(df: pd.DataFrame, k_known: int = 5, search_k: int = 20) -> pd.Series:
    # neighbor-based median imputation using BallTree (haversine)
    coords_rad = np.vstack([df["lat_rad"].values, df["lon_rad"].values]).T
    tree = BallTree(coords_rad, metric="haversine")

    pops = df["population"].values.copy()
    global_med = np.nanmedian(df["population"].values) if df["population"].notna().any() else np.nan
    country_med = df.groupby("country")["population"].transform("median").values

    n = len(df)
    for i in range(n):
        if not np.isnan(pops[i]):
            continue
        kq = min(search_k, n)
        dist, idxs = tree.query([coords_rad[i]], k=kq)
        idxs = idxs[0]
        known_vals = []
        for j in idxs:
            if j == i:
                continue
            val = pops[j]
            if not np.isnan(val):
                known_vals.append(val)
            if len(known_vals) >= k_known:
                break
        if len(known_vals) >= 1:
            pops[i] = int(np.median(known_vals))
        else:
            cm = country_med[i]
            if not np.isnan(cm):
                pops[i] = int(cm)
            elif not np.isnan(global_med):
                pops[i] = int(global_med)
            else:
                pops[i] = 0
    return pd.Series(pops).fillna(0).astype(int)

def haversine_distance_km(lat1, lon1, lat2, lon2):
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl/2)**2
    return 2 * EARTH_R * math.asin(math.sqrt(a))

@st.cache_resource(show_spinner=False)
def build_balltree(df: pd.DataFrame):
    coords_rad = np.vstack([df["lat_rad"].values, df["lon_rad"].values]).T
    tree = BallTree(coords_rad, metric="haversine")
    return tree

@st.cache_data(show_spinner=False)
def compute_nearest3(df: pd.DataFrame, tree, k_neighbors: int = 3):
    K = k_neighbors + 1
    coords_rad = np.vstack([df["lat_rad"].values, df["lon_rad"].values]).T
    distances_rad, indices = tree.query(coords_rad, k=K)
    return distances_rad, indices

def build_edges(df: pd.DataFrame, indices: np.ndarray, show_progress: bool = True) -> Dict[int, List[Tuple[int,int]]]:
    n = len(df)
    edges: Dict[int, List[Tuple[int,int]]] = {i: [] for i in range(n)}
    progress = None
    if show_progress:
        progress = st.progress(0)
    for i in range(n):
        neighbors = [j for j in indices[i] if j != i][:3]
        for rank, j in enumerate(neighbors, start=1):
            base = 2 if rank == 1 else 4 if rank == 2 else 8
            extra = 0
            if df.loc[i, "country"] != df.loc[j, "country"]:
                extra += 2
            if df.loc[j, "population_for_model"] > 200_000:
                extra += 2
            edges[i].append((j, base + extra))
        if show_progress and n > 0 and i % max(1, n // 100) == 0:
            progress.progress(int(i / n * 100))
    if show_progress:
        progress.progress(100)
    return edges

def shortest_with_path(start_city_name: str, df: pd.DataFrame, edges: Dict[int, List[Tuple[int,int]]], max_wraps: int = 10):
    matches = df.index[df["city"].str.contains(start_city_name, case=False, na=False)].tolist()
    if not matches:
        exact = df.index[df["city"] == start_city_name].tolist()
        matches = exact
    if not matches and RAPIDFUZZ_AVAILABLE:
        try:
            city_strings = (df["city"] + ", " + df["country"]).tolist()
            best = fuzzy_process.extract(start_city_name, city_strings, limit=5)
            if best:
                top_candidate = best[0][0]
                city_name = top_candidate.split(",")[0].strip()
                matches = df.index[df["city"] == city_name].tolist()
        except Exception:
            pass
    if not matches:
        return None, None
    start_idx = matches[0]

    start_state = (start_idx, 0)
    target_state = (start_idx, 1)

    pq = [(0.0, start_state)]
    dist = {start_state: 0.0}
    prev = {}

    while pq:
        cost, (u, k) = heapq.heappop(pq)
        if (u, k) == target_state:
            path = []
            node = target_state
            while node in prev:
                path.append(node)
                node = prev[node]
            path.append(start_state)
            path.reverse()
            return cost, path
        if cost > dist.get((u, k), float("inf")):
            continue
        if k > max_wraps:
            continue
        lon_u = df.loc[u, "lon_mod"]
        for v, t in edges[u]:
            lon_v = df.loc[v, "lon_mod"]
            k2 = k if lon_v > lon_u else k + 1
            if k2 > max_wraps:
                continue
            ns = (v, k2)
            new_cost = cost + t
            if new_cost < dist.get(ns, float("inf")):
                dist[ns] = new_cost
                prev[ns] = (u, k)
                heapq.heappush(pq, (new_cost, ns))
    return None, None

def reconstruct_route_df(df: pd.DataFrame, path: List[Tuple[int,int]]) -> pd.DataFrame:
    rows = []
    for step, (idx, wrap) in enumerate(path):
        rows.append({
            "step": step,
            "index": int(idx),
            "city": df.loc[idx, "city"],
            "country": df.loc[idx, "country"],
            "lat": df.loc[idx, "lat"],
            "lon": df.loc[idx, "lon"],
            "lon_mod": df.loc[idx, "lon_mod"],
            "wrap": int(wrap),
            "population": int(df.loc[idx, "population_for_model"]),
            "population_known": bool(df.loc[idx, "population_known"]),
            "country_code": df.loc[idx, "country_code"]
        })
    return pd.DataFrame(rows)

def compute_hop_times(edges: Dict[int,List[Tuple[int,int]]], path: List[Tuple[int,int]]) -> List[float]:
    times = []
    for i in range(len(path)-1):
        u = path[i][0]
        v = path[i+1][0]
        t = next((t for (x, t) in edges[u] if x == v), None)
        times.append(t if t is not None else 0)
    return times

def generate_place_ideas(route_df: pd.DataFrame, top_n: int=5) -> List[str]:
    ideas = []
    big = route_df[route_df["population"] > 1_000_000].sort_values("population", ascending=False)
    if not big.empty:
        ideas.append(f"Major cities visited: {', '.join(big['city'].unique()[:top_n])}. Good for sightseeing.")
    countries = route_df['country'].unique().tolist()
    if len(countries) > 1:
        examples = countries[:3]
        ideas.append(f"This route crosses {len(countries)} countries (e.g. {examples}). Check visas/transport.")
    lat_mean = route_df['lat'].mean()
    ideas.append(f"Average latitude of visited cities: {lat_mean:.1f}. Pack clothing accordingly.")
    small = route_df[route_df["population"] <= 200_000]
    if not small.empty:
        ideas.append("You will also visit smaller towns — great for local experiences.")
    return ideas

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Around the World — Eastward Circumnavigation", layout="wide")
st.title("🌍 Around the World — Eastward Circumnavigation")

with st.sidebar:
    st.header("Quick actions & settings")
    st.caption("Tune dataset size and solver behaviour. Use smaller samples while debugging.")
    reload = st.button("🔁 Reload dataset")

    st.markdown("### Solver depth")
    max_wraps = st.number_input(
        "Max wraps allowed (how many times you may cross the 360° longitude reset)",
        min_value=1, max_value=20, value=10, step=1
    )

    st.markdown("### City selector (UI only)")
    sample_size = st.number_input(
        "Show top N cities in dropdown (UI sample size)",
        min_value=200, max_value=20000, value=3000, step=200
    )

    st.markdown("### Data filters (speed & quality)")
    min_population = st.number_input(
        "Min city population (filter tiny places)",
        min_value=0, value=4000, step=1000
    )
    max_cities = st.number_input(
        "Max cities to keep (0 = keep all)",
        min_value=0, value=8000, step=1000
    )

    st.markdown("### Debugging")
    debug_sample_n = st.number_input(
        "DEBUG sample N (0 = off)",
        min_value=0, max_value=100000, value=0, step=1000
    )

    st.markdown("---")
    st.write("When ready:")
    run_full = st.button("▶️ Run solver")

conflicts = check_for_shadowing()
if conflicts:
    st.sidebar.warning(f"Detected local files that can shadow libraries: {conflicts}. Rename them if you have issues importing packages.")

# load dataset
try:
    if reload:
        load_and_prepare.clear()
        df = load_and_prepare(DATA_PATH, dedupe=True)
    else:
        df = load_and_prepare(DATA_PATH, dedupe=True)
except Exception as e:
    st.error(str(e))
    st.stop()

# debug sample (optional)
if debug_sample_n > 0:
    debug_n = int(min(debug_sample_n, len(df)))
    st.warning(f"DEBUG MODE: using random sample of {debug_n} rows for the entire pipeline.")
    df = df.sample(debug_n, random_state=1).reset_index(drop=True)

# apply filters
if min_population > 0:
    df = df[df["population"].fillna(0) >= min_population].reset_index(drop=True)
if max_cities > 0:
    df = df.head(max_cities).reset_index(drop=True)

with st.sidebar:
    st.markdown("---")
    st.subheader("Current configuration")
    st.write(f"- Cities loaded (after filter + limit): **{len(df):,}**")
    st.write(f"- Start city selector size: **{int(sample_size)}**")
    st.write(f"- Min population filter: **{int(min_population)}**")
    st.write(f"- Max cities to keep: **{int(max_cities)}**")
    st.write(f"- Max wraps allowed: **{int(max_wraps)}**")
    if debug_sample_n and debug_sample_n > 0:
        st.warning(f"DEBUG mode active: using sample of {int(debug_sample_n)} rows.")

# Impute population (neighbor median)
with st.spinner("Imputing missing population values (neighbor median)..."):
    try:
        df["population_for_model"] = impute_population_by_neighbor_median(df, k_known=5, search_k=20)
        df["population_known"] = ~df["population"].isna()
        st.success("Population imputation completed.")
    except Exception as e:
        st.error("Population imputation failed; falling back to zero-fill.")
        df["population_for_model"] = df["population"].fillna(0).astype(int)
        df["population_known"] = ~df["population_for_model"].isna()

# city selector values
sample_df = df.sort_values("population_for_model", ascending=False).head(int(sample_size))
city_choices = (sample_df["city"].astype(str) + ", " + sample_df["country"].astype(str)).unique().tolist()

st.sidebar.subheader("Choose start city")
start_mode = st.sidebar.radio("Input mode", ("Select from top cities", "Type name (substring match)"))
if start_mode == "Select from top cities":
    default_index = 0
    for i, v in enumerate(city_choices):
        if v.lower().startswith("london"):
            default_index = i
            break
    start_choice = st.sidebar.selectbox("Start city (City, Country)", city_choices, index=default_index)
    start_city = start_choice.split(",")[0].strip()
else:
    start_city = st.sidebar.text_input("Start city (substring)", value="London")

st.markdown("### Dataset preview")
with st.expander("Show data sample"):
    st.dataframe(df.head(200))

# build BallTree & neighbors (cached in session)
try:
    if not st.session_state["tree_built"]:
        t0 = time.time()
        with st.spinner("Building BallTree (may take a while for large datasets)..."):
            tree = build_balltree(df)
            dist_rad, indices = compute_nearest3(df, tree, k_neighbors=3)
        st.session_state["tree"] = tree
        st.session_state["dist_rad"] = dist_rad
        st.session_state["indices"] = indices
        st.session_state["tree_built"] = True
        t1 = time.time()
        st.success(f"BallTree built in {t1-t0:.1f}s for {len(df):,} cities")
    else:
        tree = st.session_state["tree"]
        dist_rad = st.session_state["dist_rad"]
        indices = st.session_state["indices"]
except Exception as e:
    st.error("Error while building BallTree. See terminal/log.")
    st.exception(e)
    st.stop()

# build edges (cached)
if st.session_state["edges"] is None:
    try:
        st.info("Building edges (this is done once per session).")
        st.session_state["edges"] = build_edges(df, indices, show_progress=True)
        st.success("Edges built and cached.")
    except Exception as e:
        st.error("Error while building edges. See terminal/log.")
        st.exception(e)
        st.stop()
else:
    edges = st.session_state["edges"]

# run solver
if run_full:
    with st.spinner("Solving shortest eastward circumnavigation..."):
        hours, path = shortest_with_path(start_city, df, st.session_state["edges"], max_wraps=int(max_wraps))
    if hours is None:
        st.error(f"Could not find a route starting from '{start_city}'. Try a different start or relax filters.")
    else:
        st.session_state["last_result"] = (float(hours), path)
        st.session_state["last_route_df"] = reconstruct_route_df(df, path)

# show last result if available
if st.session_state["last_result"] is not None:
    hours, path = st.session_state["last_result"]
    route_df = st.session_state["last_route_df"]
    days = hours / 24.0

    st.success(f"Last computed: {hours:.1f} hours ({days:.2f} days).")
    st.info("Within 80 days (1920 hours)? " + ("✅ Yes" if hours <= 1920 else "❌ No"))

    st.subheader("Route (first 50 steps)")
    st.dataframe(route_df.head(50))

    # Hop times chart (Altair)
    hop_times = compute_hop_times(st.session_state["edges"], path)
    st.subheader("Travel times per leg (hours)")
    if len(hop_times) == 0:
        st.write("No legs to display.")
    else:
        hops = []
        for i in range(len(path)-1):
            u_idx = path[i][0]
            v_idx = path[i+1][0]
            from_city = df.loc[u_idx, "city"]
            to_city = df.loc[v_idx, "city"]
            from_cc = str(df.loc[u_idx, "country_code"])
            to_cc = str(df.loc[v_idx, "country_code"])
            label = f"{from_city} ({from_cc}) → {to_city} ({to_cc})"
            hops.append({"leg": i+1, "time": hop_times[i], "from_city": from_city, "to_city": to_city, "label": label})
        hop_df = pd.DataFrame(hops)

        base = (
            alt.Chart(hop_df)
            .mark_bar(color="#7fb3ff")
            .encode(
                x=alt.X("label:O", title="Leg (From → To)", sort=None, axis=alt.Axis(labelAngle=-40)),
                y=alt.Y("time:Q", title="Travel time (hours)"),
                tooltip=[alt.Tooltip("leg:O", title="Leg #"), alt.Tooltip("label:N", title="From → To"), alt.Tooltip("time:Q", title="Hours")]
            )
            .properties(width="container", height=380)
        )

        text = base.mark_text(dy=-10, color="white", fontSize=12).encode(text=alt.Text("time:Q", format=".0f"))
        layered = alt.layer(base, text).configure_axis(grid=False)
        st.altair_chart(layered, use_container_width=True)

    # ---------------------------
    # Matplotlib longitude progression
    # ---------------------------
    st.subheader("Longitude (mod 360) progression (blue = east, red = west)")

    lon_vals = route_df["lon_mod"].values
    wraps = route_df["wrap"].values
    steps = route_df["step"].values
    n_points = len(lon_vals)

    seg_lines = []
    seg_colors = []
    westward_info = []

    for i in range(n_points - 1):
        lon_i = float(lon_vals[i])
        lon_j = float(lon_vals[i+1])
        wrap_i = int(wraps[i])
        wrap_j = int(wraps[i+1])
        if wrap_j > wrap_i:
            direction = "east"
        elif lon_j > lon_i:
            direction = "east"
        else:
            direction = "west"
        x0, x1 = float(steps[i]), float(steps[i+1])
        y0, y1 = lon_i, lon_j
        seg_lines.append([(x0, y0), (x1, y1)])
        seg_colors.append("#4da6ff" if direction == "east" else "#ff6b6b")
        if direction == "west":
            from_city = route_df.loc[i, "city"]
            to_city = route_df.loc[i+1, "city"]
            westward_info.append({"leg": i+1, "from_to": f"{from_city} → {to_city}", "step_i": int(steps[i]), "step_j": int(steps[i+1])})

    # Apply dark background style
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 3.2))
    if seg_lines:
        lc = LineCollection(seg_lines, colors=seg_colors, linewidths=3, zorder=2)
        ax.add_collection(lc)
    ax.plot(steps, lon_vals, marker='o', linestyle='None', markersize=6, color='white', zorder=3)
    if len(steps) > 0:
        ax.set_xlim(min(steps) - 0.5, max(steps) + 0.5)
    else:
        ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 360)
    ax.set_ylabel("Longitude (0–360°)", color="white")
    ax.set_xlabel("Step", color="white")
    ax.set_title("Longitude progression (blue = east, red = west)", color="white")
    ax.grid(color="#444444", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig)

    # show warnings if westward segments exist
    if len(westward_info) > 0:
        st.warning(f"⚠️ Westward movement detected on {len(westward_info)} leg(s). Eastward circumnavigation should minimize westward hops.")
        st.table(pd.DataFrame(westward_info).head(20))
    else:
        st.success("No westward movements detected — route is eastward (modulo wraps).")

    # Population plot (log scale)
    st.subheader("Population of visited cities (log scale)")
    try:
        st.line_chart(np.log10(route_df["population"].replace(0, np.nan).fillna(0) + 1))
    except Exception:
        st.line_chart(route_df["population"])

    st.markdown("**Route summary**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total hours", f"{hours:.1f}")
    c2.metric("Total days", f"{days:.2f}")
    c3.metric("Within 80 days", "Yes" if hours <= 1920 else "No")

    # save results
    try:
        summary = {"start_city": start_city, "hours": float(hours), "days": float(days), "within_80_days": hours <= 1920}
        with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        route_df.to_csv(os.path.join(RESULTS_DIR, "route_states.csv"), index=False)
        st.success(f"Results saved to `{RESULTS_DIR}/`")
    except Exception:
        st.warning("Could not save results to disk (check permissions).")

    # travel ideas
    st.subheader("Travel ideas & tips")
    ideas = generate_place_ideas(route_df)
    for idea in ideas:
        st.write("- " + idea)

    # folium map (highlight imputed populations)
    st.subheader("Interactive route map (imputed populations highlighted)")
    m = folium.Map(location=[20,0], zoom_start=2)
    coords = []
    for (idx, w) in path:
        lat = df.loc[idx, "lat"]; lon = df.loc[idx, "lon"]
        coords.append((lat, lon))
        popup_html = f"{df.loc[idx,'city']} ({df.loc[idx,'country']})<br>wrap={w}<br>population={int(df.loc[idx,'population_for_model']):,}"
        known_flag = bool(df.loc[idx, "population_known"])
        if known_flag:
            folium.CircleMarker([lat, lon], radius=4, color="blue", fill=True, fill_opacity=0.8, popup=popup_html).add_to(m)
        else:
            folium.CircleMarker([lat, lon], radius=6, color="red", fill=True, fill_opacity=0.9,
                                popup=popup_html + "<br><b>population imputed</b>").add_to(m)
    AntPath(locations=coords, weight=3).add_to(m)

    # small legend
    legend_html = """
     <div style="position: fixed; 
                 bottom: 50px; left: 10px; width:180px; height:80px; 
                 background-color: rgba(0,0,0,0.6); z-index:9999; font-size:12px;
                 color: white; padding:8px; border-radius:6px;">
     <b>Legend</b><br>
     <span style="color:blue;">●</span> population known<br>
     <span style="color:red;">●</span> population imputed<br>
     <span style="color:cornflowerblue;">—</span> longitude east segment<br>
     <span style="color:salmon;">—</span> longitude west segment
     </div>
     """
    m.get_root().html.add_child(folium.Element(legend_html))
    st_folium(m, width=900, height=600)

st.markdown("---")
st.caption("Tip: use DEBUG sample N / min population / max cities to speed debugging. Click Reload dataset if you changed the CSV.")
