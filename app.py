"""
Orbital City Lab — Signal-Resilient Transit Tracker
====================================================

Public transit vehicles keep sending GPS over the mobile network. During
electronic warfare (REB) activity that network can drop out for seconds at
a time. This app simulates a small bus/tram fleet on a real Kyiv route and
shows what riders would see on a live map:

  * green solid GPS fix — position comes straight from the "vehicle"
  * red dead reckoning — GPS is lost, so position is estimated from a
    trained speed model + route geometry until the signal comes back

Everything (fleet state, physics step, ML inference) runs inside this one
Streamlit session — no background sockets, no shared files on disk. That is
what makes it safe to deploy as a single public app on Streamlit Community
Cloud, where every visitor gets an independent in-memory simulation.
"""

import time

import joblib
import numpy as np
import pydeck as pdk
import streamlit as st

import feature_utils
import route_utils as ru

# ==========================================================================
# CONSTANTS
# ==========================================================================
TICK_SEC = 0.5                    # simulation step length (seconds of sim-time)
REFRESH_SEC = 0.5                 # how often the fragment redraws

REB_CYCLE_SEC = 20                # full online+offline cycle per vehicle
REB_OFFLINE_START = 6             # signal drops at this point in the cycle
REB_OFFLINE_END = 14              # signal returns at this point in the cycle

MODEL_SAMPLE_INTERVAL_SEC = 15.0  # matches training step of model.pkl
MODEL_HISTORY_LEN = 5

SPEED_NOISE_STD_KMH = 0.9
SPEED_NOISE_DECAY = 0.85
SPEED_NOISE_CLAMP_KMH = 3.0

VEHICLE_NAMES = ["Bus 101", "Bus 204", "Bus 170", "Bus 212", "Bus 318"]

st.set_page_config(
    page_title="REBoot — трекер громадського транспорту",
    page_icon="📡",
    layout="wide",
)

# ==========================================================================
# STYLE — dark, "signal-resilience" visual language for REBoot
# ==========================================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

    .stApp {
        background:
            radial-gradient(1200px 600px at 15% -10%, rgba(61,217,196,0.08), transparent 60%),
            radial-gradient(1000px 500px at 100% 0%, rgba(56,189,248,0.06), transparent 55%),
            #0A0E17;
    }

    section[data-testid="stSidebar"] {
        background: #0D1220;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.6rem; }

    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }

    /* ---- brand header ---- */
    .reboot-header {
        display: flex;
        align-items: center;
        gap: 16px;
        padding: 4px 0 6px 0;
        margin-bottom: 4px;
    }
    .reboot-logo {
        width: 52px; height: 52px; border-radius: 14px;
        display: flex; align-items: center; justify-content: center;
        font-size: 26px;
        background: linear-gradient(135deg, #3DD9C4 0%, #2563EB 100%);
        box-shadow: 0 0 24px rgba(61,217,196,0.35);
        flex-shrink: 0;
    }
    .reboot-title { font-size: 30px; font-weight: 800; color: #F3F6FB; letter-spacing: -0.02em; line-height: 1.1; }
    .reboot-title span { color: #3DD9C4; }
    .reboot-subtitle { font-size: 14.5px; color: #8B96AB; margin-top: 2px; }
    .reboot-badge {
        margin-left: auto; align-self: flex-start;
        background: rgba(61,217,196,0.12); color: #3DD9C4;
        border: 1px solid rgba(61,217,196,0.35);
        font-size: 11.5px; font-weight: 700; letter-spacing: 0.06em;
        padding: 5px 10px; border-radius: 999px; text-transform: uppercase;
        white-space: nowrap;
    }

    /* ---- legend strip ---- */
    .reboot-legend {
        display: flex; gap: 22px; flex-wrap: wrap;
        padding: 10px 16px; margin: 10px 0 18px 0;
        background: #121826; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px; font-size: 13px; color: #C4CCDB;
    }
    .reboot-legend b { color: #F3F6FB; }
    .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; }

    /* ---- section labels ---- */
    .reboot-section-label {
        font-size: 12px; font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: #6C7890; margin: 2px 0 10px 2px;
    }

    /* ---- vehicle cards ---- */
    .vcard {
        border-radius: 14px; padding: 14px 16px; margin-bottom: 12px;
        background: #121826; border: 1px solid rgba(255,255,255,0.06);
        border-left: 3px solid var(--accent, #3DD9C4);
        transition: border-color .2s ease;
    }
    .vcard-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
    .vcard-name { font-weight: 700; font-size: 15px; color: #F3F6FB; }
    .vcard-name .vicon { margin-right: 8px; opacity: 0.85; }
    .vcard-pill {
        font-size: 11px; font-weight: 700; letter-spacing: 0.03em;
        padding: 3px 9px; border-radius: 999px; white-space: nowrap;
    }
    .pill-live { background: rgba(52,211,153,0.14); color: #34D399; border: 1px solid rgba(52,211,153,0.35); }
    .pill-lost { background: rgba(251,113,133,0.14); color: #FB7185; border: 1px solid rgba(251,113,133,0.4); animation: pulse 1.4s ease-in-out infinite; }
    .pill-done { background: rgba(139,150,171,0.14); color: #8B96AB; border: 1px solid rgba(139,150,171,0.3); }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.55; } }

    .vcard-speed { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 600; color: #E7ECF5; }
    .vcard-speed span { font-size: 12px; font-weight: 500; color: #6C7890; margin-left: 4px; }

    hr, div[data-testid="stDivider"] { border-color: rgba(255,255,255,0.08) !important; }

    .stButton > button {
        border-radius: 10px !important; font-weight: 600 !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==========================================================================
# CACHED RESOURCES (loaded once per server process, shared read-only)
# ==========================================================================
@st.cache_resource(show_spinner="Loading speed-prediction model...")
def load_model():
    try:
        return joblib.load("model.pkl")
    except Exception as e:
        st.warning(f"Could not load model.pkl, falling back to route geometry only: {e}")
        return None


@st.cache_data(show_spinner="Loading route...")
def load_route():
    return ru.load_or_build_route()


ai_model = load_model()
route_coords = load_route()
ROUTE_LEN = len(route_coords)


# ==========================================================================
# SIMULATION — one simulated vehicle's worth of state + physics
# ==========================================================================
def new_vehicle(name, start_offset_idx, phase_offset_sec):
    idx = min(start_offset_idx, ROUTE_LEN - 2)
    lat, lon = route_coords[idx]
    return {
        "name": name,
        "idx": idx,
        "lat": lat,
        "lon": lon,
        "speed": ru.BASE_SPEED_KMH,
        "speed_noise": 0.0,
        "heading": ru.heading_for_route_idx(route_coords, idx),
        "speed_buffer": [],
        "model_speed_window": [],
        "model_heading_window": [],
        "next_model_sample_time": phase_offset_sec,
        "elapsed": phase_offset_sec,
        "phase_offset": phase_offset_sec,
        "reb_interp": None,
        "is_predicted": False,
        "reached_end": False,
    }


def is_signal_lost(elapsed_sec):
    phase = elapsed_sec % REB_CYCLE_SEC
    return REB_OFFLINE_START <= phase <= REB_OFFLINE_END


def start_reb_prediction(v, current_time):
    """Called once at the moment signal is lost, mirrors server_core.py logic."""
    last_speed = v["speed_buffer"][-1] if v["speed_buffer"] else ru.BASE_SPEED_KMH
    speed_window = v["model_speed_window"]
    heading_window = v["model_heading_window"]

    geometric_target = ru.target_speed_for_position(route_coords, v["idx"])

    if len(speed_window) >= MODEL_HISTORY_LEN and ai_model is not None:
        try:
            features = feature_utils.build_features(speed_window, heading_window)
            delta = float(ai_model.predict([features])[0])
            model_target = speed_window[-1] + delta
            target_speed = 0.5 * model_target + 0.5 * geometric_target
        except Exception:
            target_speed = geometric_target
    else:
        target_speed = geometric_target

    return {"start_time": current_time, "start_speed": last_speed, "target_speed": target_speed}


def sample_interp_speed(interp, current_time):
    elapsed = current_time - interp["start_time"]
    ratio = min(1.0, elapsed / MODEL_SAMPLE_INTERVAL_SEC)
    return interp["start_speed"] + ratio * (interp["target_speed"] - interp["start_speed"])


def step_vehicle(v, dt):
    if v["reached_end"]:
        return v

    v["elapsed"] += dt
    signal_lost = is_signal_lost(v["elapsed"])

    if not signal_lost:
        # --- normal GPS-driven movement (mirrors the old client_sim.py) ---
        clean_target = ru.target_speed_for_position(route_coords, v["idx"])
        v["speed_noise"] = v["speed_noise"] * SPEED_NOISE_DECAY + np.random.normal(0.0, SPEED_NOISE_STD_KMH)
        v["speed_noise"] = max(-SPEED_NOISE_CLAMP_KMH, min(SPEED_NOISE_CLAMP_KMH, v["speed_noise"]))
        noisy_target = max(0.0, clean_target + v["speed_noise"])
        v["speed"] = ru.step_speed_toward(v["speed"], noisy_target, dt_sec=dt)
        v["heading"] = ru.heading_for_route_idx(route_coords, v["idx"])

        speed_mps = v["speed"] * (1000 / 3600)
        v["lat"], v["lon"], v["idx"], reached = ru.advance_along_route(
            route_coords, v["idx"], v["lat"], v["lon"], speed_mps * dt
        )
        v["reached_end"] = reached
        v["is_predicted"] = False
        v["reb_interp"] = None

        v["speed_buffer"].append(v["speed"])
        v["speed_buffer"] = v["speed_buffer"][-5:]

        if v["elapsed"] >= v["next_model_sample_time"]:
            v["model_speed_window"].append(v["speed"])
            v["model_heading_window"].append(v["heading"])
            v["model_speed_window"] = v["model_speed_window"][-MODEL_HISTORY_LEN:]
            v["model_heading_window"] = v["model_heading_window"][-MODEL_HISTORY_LEN:]
            v["next_model_sample_time"] = v["elapsed"] + MODEL_SAMPLE_INTERVAL_SEC

    else:
        # --- signal lost: dead reckoning (mirrors the old server_core.py) ---
        if v["reb_interp"] is None:
            v["reb_interp"] = start_reb_prediction(v, v["elapsed"])

        model_speed_now = sample_interp_speed(v["reb_interp"], v["elapsed"])
        last_actual_speed = v["speed_buffer"][-1] if v["speed_buffer"] else ru.BASE_SPEED_KMH
        predicted_speed = max(15.0, ru.step_speed_toward(last_actual_speed, model_speed_now, dt_sec=dt))

        speed_mps = predicted_speed * (1000 / 3600)
        new_lat, new_lon, new_idx, reached = ru.advance_along_route(
            route_coords, v["idx"], v["lat"], v["lon"], speed_mps * dt
        )
        if new_idx != v["idx"]:
            v["heading"] = ru.heading_for_route_idx(route_coords, new_idx)

        v["lat"], v["lon"], v["idx"] = new_lat, new_lon, new_idx
        v["reached_end"] = reached
        v["is_predicted"] = True
        v["speed"] = predicted_speed

        v["speed_buffer"].append(predicted_speed)
        v["speed_buffer"] = v["speed_buffer"][-5:]

    return v


# ==========================================================================
# SESSION STATE
# ==========================================================================
def init_fleet(num_vehicles):
    spread = max(1, (ROUTE_LEN - 2) // max(1, num_vehicles))
    fleet = {}
    for i in range(num_vehicles):
        name = VEHICLE_NAMES[i % len(VEHICLE_NAMES)]
        fleet[name] = new_vehicle(
            name,
            start_offset_idx=i * spread,
            phase_offset_sec=i * (REB_CYCLE_SEC / max(1, num_vehicles)),
        )
    return fleet


if "num_vehicles" not in st.session_state:
    st.session_state.num_vehicles = 2
if "fleet" not in st.session_state:
    st.session_state.fleet = init_fleet(st.session_state.num_vehicles)
if "running" not in st.session_state:
    st.session_state.running = True
if "speed_multiplier" not in st.session_state:
    st.session_state.speed_multiplier = 1.0
if "last_tick_wall_time" not in st.session_state:
    st.session_state.last_tick_wall_time = time.time()


# ==========================================================================
# SIDEBAR — controls
# ==========================================================================
with st.sidebar:
    st.markdown(
        """
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:18px;">
            <div style="width:34px; height:34px; border-radius:10px; display:flex; align-items:center;
                        justify-content:center; font-size:17px;
                        background:linear-gradient(135deg,#3DD9C4 0%,#2563EB 100%);">📡</div>
            <div style="font-weight:800; font-size:17px; color:#F3F6FB;">
                <span style="color:#F3F6FB;">REB</span><span style="color:#3DD9C4;">oot</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="reboot-section-label">Керування симуляцією</div>', unsafe_allow_html=True)

    if st.button(
        "▶️  Відновити" if not st.session_state.running else "⏸️  Пауза",
        use_container_width=True,
    ):
        st.session_state.running = not st.session_state.running

    if st.button("🔄  Скинути симуляцію", use_container_width=True):
        st.session_state.fleet = init_fleet(st.session_state.num_vehicles)
        st.session_state.running = True

    st.write("")
    num_vehicles = st.slider("Розмір парку", min_value=1, max_value=5, value=st.session_state.num_vehicles)
    if num_vehicles != st.session_state.num_vehicles:
        st.session_state.num_vehicles = num_vehicles
        st.session_state.fleet = init_fleet(num_vehicles)

    st.session_state.speed_multiplier = st.slider(
        "Швидкість відтворення", min_value=0.5, max_value=4.0, value=st.session_state.speed_multiplier, step=0.5,
        help="Прискорює хід симуляції (на реалістичність фізичної моделі не впливає).",
    )

    st.divider()
    st.markdown('<div class="reboot-section-label">Умовні позначення</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div style="font-size:13px; line-height:1.6; color:#C4CCDB;">
            <span class="dot" style="background:#34D399;"></span>
            <b style="color:#F3F6FB;">GPS-сигнал стабільний</b> — позиція надходить напряму від транспорту.<br><br>
            <span class="dot" style="background:#FB7185;"></span>
            <b style="color:#F3F6FB;">Втрата сигналу (РЕБ)</b> — позиція оцінюється моделлю швидкості
            та геометрією маршруту, доки GPS не відновиться.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption(
        "Це автономна симуляція для демонстрації: реальних транспортних засобів, "
        "обладнання РЕБ чи персональних даних тут немає."
    )


# ==========================================================================
# MAIN CONTENT (auto-refreshing fragment)
# ==========================================================================
st.markdown(
    """
    <div class="reboot-header">
        <div class="reboot-logo">📡</div>
        <div>
            <div class="reboot-title" style="color:#3DD9C4;"><span style="color:#F3F6FB;">REB</span>oot</div>
            <div class="reboot-subtitle">Транспорт на карті — навіть коли глушать GPS</div>
        </div>
        <div class="reboot-badge">MVP · Демо</div>
    </div>
    <div class="reboot-legend">
        <div><span class="dot" style="background:#34D399;"></span><b>GPS стабільний</b> — реальна позиція</div>
        <div><span class="dot" style="background:#FB7185;"></span><b>Втрата сигналу</b> — оцінка позиції під час РЕБ</div>
        <div><span class="dot" style="background:#8B96AB;"></span><b>Маршрут завершено</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.fragment(run_every=REFRESH_SEC)
def render():
    now = time.time()
    real_dt = now - st.session_state.last_tick_wall_time
    st.session_state.last_tick_wall_time = now
    real_dt = max(0.0, min(real_dt, 2.0))

    if st.session_state.running:
        sim_dt = real_dt * st.session_state.speed_multiplier
        remaining = sim_dt
        while remaining > 0:
            step = min(TICK_SEC, remaining)
            for v in st.session_state.fleet.values():
                step_vehicle(v, step)
            remaining -= step

    fleet = st.session_state.fleet
    all_done = all(v["reached_end"] for v in fleet.values())

    col_stats, col_map = st.columns([1, 2])

    with col_stats:
        st.markdown('<div class="reboot-section-label">Стан парку</div>', unsafe_allow_html=True)

        def vehicle_icon(name):
            n = name.lower()
            if "tram" in n:
                return "🚊"
            if "trolley" in n:
                return "🚎"
            return "🚌"

        for v in fleet.values():
            if v["reached_end"]:
                pill_cls, pill_text, accent = "pill-done", "🏁 Маршрут завершено", "#8B96AB"
            elif v["is_predicted"]:
                pill_cls, pill_text, accent = "pill-lost", "🔴 Сигнал втрачено", "#FB7185"
            else:
                pill_cls, pill_text, accent = "pill-live", "🟢 GPS стабільний", "#34D399"

            st.markdown(
                f"""
                <div class="vcard" style="--accent:{accent};">
                    <div class="vcard-top">
                        <div class="vcard-name"><span class="vicon">{vehicle_icon(v['name'])}</span>{v['name']}</div>
                        <div class="vcard-pill {pill_cls}">{pill_text}</div>
                    </div>
                    <div class="vcard-speed">{v['speed']:.1f}<span>км/год</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if all_done:
            st.success("Усі транспортні засоби дісталися кінця маршруту.")

    with col_map:
        st.markdown('<div class="reboot-section-label">Карта маршруту</div>', unsafe_allow_html=True)

        SIGNAL_GREEN = [45, 212, 158]
        SIGNAL_RED = [251, 113, 133]
        ROUTE_BLUE = [56, 189, 248]  # Яскравий синій колір для маршруту

        layers = []

        # 1. Шар самого маршруту (підсвічений синім)
        path_layer = pdk.Layer(
            "PathLayer",
            data=[{"path": [[p[1], p[0]] for p in route_coords]}],
            get_path="path",
            get_color=ROUTE_BLUE + [200],  
            width_scale=20,
            width_min_pixels=4,  
        )
        layers.append(path_layer)

        # 2. Кінцеві точки (менші, у колір маршруту)
        endpoints = [
            {"coord": [route_coords[0][1], route_coords[0][0]], "color": ROUTE_BLUE + [255]},
            {"coord": [route_coords[-1][1], route_coords[-1][0]], "color": ROUTE_BLUE + [180]},
        ]
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=endpoints,
            get_position="coord",
            get_fill_color="color",
            get_radius=25,       
            radius_min_pixels=4,
        ))

        # 3. М'яке світіння (РЕБ) за автобусами
        glow_list = [
            {"coord": [v["lon"], v["lat"]], "color": SIGNAL_RED + [70]}
            for v in fleet.values() if v["is_predicted"] and not v["reached_end"]
        ]
        if glow_list:
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=glow_list,
                get_position="coord",
                get_fill_color="color",
                get_radius=140,
                radius_min_pixels=22,
            ))

        # 4. Класичні круглі точки для самих машин (як були в оригіналі)
        v_list = []
        for v in fleet.values():
            if not v["reached_end"]:
                color = (SIGNAL_RED if v["is_predicted"] else SIGNAL_GREEN) + [255]
                v_list.append({
                    "coord": [v["lon"], v["lat"]],
                    "color": color,
                    "tooltip": f"{v['name']} — {v['speed']:.0f} км/год"
                               f"{' · сигнал втрачено' if v['is_predicted'] else ''}",
                })
        layers.append(pdk.Layer(
            "ScatterplotLayer",
            data=v_list,
            get_position="coord",
            get_fill_color="color",
            get_line_color=[10, 14, 23, 255],
            stroked=True,
            line_width_min_pixels=2,
            get_radius=60,
            radius_min_pixels=10,
        ))

        first = next(iter(fleet.values()))
        view_state = pdk.ViewState(latitude=first["lat"], longitude=first["lon"], zoom=14, pitch=30)

        st.pydeck_chart(
            pdk.Deck(
                layers=layers,
                initial_view_state=view_state,
                tooltip={
                    "text": "{tooltip}",
                    "style": {"backgroundColor": "#121826", "color": "#E7ECF5", "fontSize": "13px"},
                },
                map_style="dark",
            ),
            use_container_width=True,
        )


render()
