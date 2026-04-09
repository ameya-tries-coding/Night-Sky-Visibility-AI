"""
Hybrid Celestial Visibility Prediction System
==============================================
Frontend redesign — all backend logic preserved exactly.
Only UI/UX structure, layout, and styling has been changed.
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict
from src.models.ranking import rank_objects
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import requests
from datetime import datetime, date, time
import math
# -------------------------------
# BACKEND IMPORTS (ADD THIS)
# -------------------------------
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.predict import predict
from src.models.ranking import rank_objects

# If these are inside app.py earlier, copy them here instead
# Otherwise import if you moved them to files

# OPTIONAL (only if you moved to astro_engine.py)
# from src.astronomy.astro_engine import get_real_objects
# ── Keep ALL backend imports exactly as they were in your original file ──────
# from skyfield.api import load, Topos, Star
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
# import joblib
# ... etc.

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Celestial Vision | AI Sky Predictor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — dark space aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;800&family=Syne:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root tokens ── */
:root {
    --bg-void:        #020408;
    --bg-deep:        #080d14;
    --bg-card:        #0d1520;
    --bg-card-hover:  #111d2e;
    --border:         rgba(96, 165, 250, 0.12);
    --border-bright:  rgba(96, 165, 250, 0.35);
    --accent-blue:    #60a5fa;
    --accent-cyan:    #22d3ee;
    --accent-gold:    #fbbf24;
    --accent-green:   #34d399;
    --accent-red:     #f87171;
    --accent-purple:  #a78bfa;
    --text-primary:   #e2e8f0;
    --text-secondary: #94a3b8;
    --text-dim:       #4b6070;
    --glow-blue:      0 0 20px rgba(96,165,250,0.25);
    --glow-cyan:      0 0 20px rgba(34,211,238,0.25);
    --glow-gold:      0 0 20px rgba(251,191,36,0.3);
}

/* ── App shell ── */
.stApp {
    background: var(--bg-void);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(96,165,250,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 40% 30% at 80% 110%, rgba(167,139,250,0.06) 0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='noise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23noise)' opacity='0.03'/%3E%3C/svg%3E");
    font-family: 'Syne', sans-serif;
    color: var(--text-primary);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1400px; }
.stDeployButton { display: none; }

/* ── Headings ── */
h1, h2, h3 { font-family: 'Orbitron', monospace; }

/* ── Dividers ── */
hr { border-color: var(--border); margin: 2rem 0; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stTimeInput > div > div > input {
    background: rgba(13,21,32,0.9) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-family: 'Syne', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus,
.stDateInput > div > div > input:focus,
.stTimeInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: var(--glow-blue) !important;
}

/* ── Labels ── */
.stTextInput label, .stDateInput label, .stTimeInput label,
.stSelectbox label { 
    color: var(--text-secondary) !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(96,165,250,0.15), rgba(34,211,238,0.10)) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: 10px !important;
    color: var(--accent-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.25s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, rgba(96,165,250,0.28), rgba(34,211,238,0.22)) !important;
    box-shadow: 0 0 24px rgba(34,211,238,0.3), 0 0 8px rgba(96,165,250,0.2) !important;
    transform: translateY(-1px) !important;
    border-color: var(--accent-cyan) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Dataframe / table ── */
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stDataFrame thead th {
    background: rgba(96,165,250,0.08) !important;
    color: var(--accent-blue) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(96,165,250,0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(96,165,250,0.45); }

/* ── Matplotlib figure background ── */
.stPlotlyChart, .stPyplot { border-radius: 14px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def card(content_fn, title=None, accent="blue", extra_css=""):
    """Renders a floating dark card.  Pass a callable that emits Streamlit widgets."""
    accent_map = {
        "blue":   ("var(--accent-blue)",  "rgba(96,165,250,0.08)"),
        "cyan":   ("var(--accent-cyan)",  "rgba(34,211,238,0.08)"),
        "gold":   ("var(--accent-gold)",  "rgba(251,191,36,0.08)"),
        "green":  ("var(--accent-green)", "rgba(52,211,153,0.08)"),
        "red":    ("var(--accent-red)",   "rgba(248,113,113,0.08)"),
        "purple": ("var(--accent-purple)","rgba(167,139,250,0.08)"),
    }
    color, bg = accent_map.get(accent, accent_map["blue"])
    header_html = f"""
        <div style='font-family:Orbitron,monospace;font-size:0.7rem;font-weight:600;
                    letter-spacing:0.14em;text-transform:uppercase;color:{color};
                    margin-bottom:0.9rem;display:flex;align-items:center;gap:0.5rem;'>
            <span style='width:3px;height:14px;background:{color};
                         border-radius:2px;display:inline-block;'></span>
            {title}
        </div>
    """ if title else ""
    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color}22;
                border-radius:14px;padding:1.4rem 1.6rem;
                margin-bottom:1rem;{extra_css}'>
        {header_html}
    """, unsafe_allow_html=True)
    content_fn()
    st.markdown("</div>", unsafe_allow_html=True)


def metric_pill(label, value, unit="", accent="blue", icon=""):
    accent_map = {
        "blue":   ("#60a5fa", "rgba(96,165,250,0.12)"),
        "cyan":   ("#22d3ee", "rgba(34,211,238,0.12)"),
        "gold":   ("#fbbf24", "rgba(251,191,36,0.12)"),
        "green":  ("#34d399", "rgba(52,211,153,0.12)"),
        "red":    ("#f87171", "rgba(248,113,113,0.12)"),
        "purple": ("#a78bfa", "rgba(167,139,250,0.12)"),
    }
    color, bg = accent_map.get(accent, accent_map["blue"])
    st.markdown(f"""
    <div style='background:{bg};border:1px solid {color}33;border-radius:12px;
                padding:1rem 1.2rem;text-align:center;height:100%;'>
        <div style='font-size:1.4rem;margin-bottom:0.3rem;'>{icon}</div>
        <div style='font-family:Orbitron,monospace;font-size:1.25rem;font-weight:700;
                    color:{color};line-height:1.1;'>
            {value}<span style='font-size:0.75rem;font-weight:400;
                                 color:{color};opacity:0.7;margin-left:2px;'>{unit}</span>
        </div>
        <div style='font-size:0.7rem;font-weight:600;letter-spacing:0.1em;
                    text-transform:uppercase;color:#64748b;margin-top:0.35rem;'>
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_divider(label):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:1rem;margin:2.2rem 0 1.4rem;'>
        <div style='flex:1;height:1px;background:linear-gradient(90deg,
             transparent,rgba(96,165,250,0.3),transparent);'></div>
        <span style='font-family:Orbitron,monospace;font-size:0.65rem;font-weight:600;
                     letter-spacing:0.2em;text-transform:uppercase;color:#4b6070;
                     white-space:nowrap;'>{label}</span>
        <div style='flex:1;height:1px;background:linear-gradient(90deg,
             transparent,rgba(96,165,250,0.3),transparent);'></div>
    </div>
    """, unsafe_allow_html=True)


def aqi_badge(aqi_value):
    """Color-coded AQI badge."""
    if aqi_value is None:
        return "—", "blue"
    if aqi_value <= 50:
        return f"{aqi_value} · Good", "green"
    elif aqi_value <= 100:
        return f"{aqi_value} · Moderate", "gold"
    elif aqi_value <= 150:
        return f"{aqi_value} · Unhealthy", "gold"
    elif aqi_value <= 200:
        return f"{aqi_value} · Unhealthy+", "red"
    else:
        return f"{aqi_value} · Hazardous", "red"


def visibility_alert(score):
    """Returns (label, color_key, emoji) based on visibility score."""
    if score >= 75:
        return "Excellent Conditions", "green", "✦"
    elif score >= 45:
        return "Moderate Conditions", "gold", "◈"
    else:
        return "Poor Conditions", "red", "✕"


def object_icon(obj_type: str) -> str:
    t = str(obj_type).lower()
    if "planet" in t:   return "🌍"
    if "star" in t:     return "⭐"
    if "constell" in t: return "🌌"
    if "moon" in t:     return "🌕"
    if "galaxy" in t:   return "🌀"
    return "✦"


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════  HEADER  ═════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding:2.5rem 0 1rem;text-align:center;position:relative;'>
    <!-- Star field decorations -->
    <div style='position:absolute;top:1rem;left:5%;font-size:0.5rem;
                color:rgba(96,165,250,0.4);letter-spacing:0.8rem;'>
        · · · · · · · ·
    </div>
    <div style='position:absolute;top:1rem;right:5%;font-size:0.5rem;
                color:rgba(96,165,250,0.4);letter-spacing:0.8rem;'>
        · · · · · · · ·
    </div>

    <!-- Eyepiece ring -->
    <div style='display:inline-block;width:64px;height:64px;
                border-radius:50%;border:2px solid rgba(34,211,238,0.4);
                box-shadow:0 0 30px rgba(34,211,238,0.2),inset 0 0 20px rgba(34,211,238,0.05);
                display:flex;align-items:center;justify-content:center;
                font-size:1.8rem;margin:0 auto 1rem;line-height:64px;'>
        🔭
    </div>

    <h1 style='font-family:Orbitron,monospace;font-size:clamp(1.4rem,3.5vw,2.4rem);
               font-weight:800;letter-spacing:0.08em;
               background:linear-gradient(135deg,#e2e8f0 0%,#60a5fa 50%,#22d3ee 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;margin:0 0 0.5rem;line-height:1.2;'>
        HYBRID CELESTIAL VISIBILITY
    </h1>
    <h2 style='font-family:Orbitron,monospace;font-size:clamp(0.75rem,1.8vw,1.05rem);
               font-weight:400;letter-spacing:0.28em;color:#4b6070;margin:0 0 0.8rem;'>
        PREDICTION SYSTEM
    </h2>
    <p style='font-family:Syne,sans-serif;font-size:0.88rem;color:#64748b;
              max-width:560px;margin:0 auto;line-height:1.7;'>
        AI-powered sky analysis combining real-time weather, atmospheric quality,
        lunar phase, and orbital mechanics to rank tonight's best celestial targets.
    </p>

    <!-- Thin rule with dots -->
    <div style='display:flex;align-items:center;justify-content:center;
                gap:1rem;margin-top:2rem;'>
        <div style='width:80px;height:1px;background:linear-gradient(90deg,transparent,rgba(96,165,250,0.4));'></div>
        <span style='color:rgba(96,165,250,0.5);font-size:0.45rem;letter-spacing:0.6rem;'>◆ ◆ ◆</span>
        <div style='width:80px;height:1px;background:linear-gradient(90deg,rgba(96,165,250,0.4),transparent);'></div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════  INPUT PANEL  ════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────
section_divider("Observation Parameters")

with st.container():
    st.markdown("""
    <div style='background:linear-gradient(135deg,rgba(13,21,32,0.95),rgba(8,13,20,0.95));
                border:1px solid rgba(96,165,250,0.18);border-radius:16px;
                padding:1.8rem 2rem;margin-bottom:1.5rem;
                box-shadow:0 4px 40px rgba(0,0,0,0.5);'>
        <div style='font-family:Orbitron,monospace;font-size:0.65rem;font-weight:600;
                    letter-spacing:0.2em;text-transform:uppercase;color:#4b6070;
                    margin-bottom:1.4rem;'>
            ◈ &nbsp; Enter Location &amp; Time
        </div>
    """, unsafe_allow_html=True)

    col_loc, col_date, col_time, col_btn = st.columns([3, 2, 2, 1.4])

    with col_loc:
        city = st.text_input(
            "📍 City / Location",
            placeholder="e.g. Mumbai, Pune, New York…",
            help="Enter any city name — coordinates are fetched automatically via geocoding."
        )

    with col_date:
        obs_date = st.date_input(
            "📅 Observation Date",
            value=date.today(),
            help="Select the date of your observation session."
        )

    with col_time:
        obs_time = st.time_input(
            "🕐 Local Time",
            value=time(21, 0),
            help="Choose your local observation start time."
        )

    with col_btn:
        st.markdown("<div style='height:1.95rem'></div>", unsafe_allow_html=True)
        run_button = st.button("⚡ Analyze Sky", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════  BACKEND PREDICTION  ═════════════════════════════════
# This block calls your existing predict() function.
# Replace the placeholder logic below with your actual backend call.
# ─────────────────────────────────────────────────────────────────────────────

# ── Placeholder result structure that mirrors what your backend would return ──
# IMPORTANT: swap the body of `run_prediction()` for your real predict() call.

def run_prediction(city, obs_date, obs_time):

    # Convert time → hour
    hour = obs_time.hour

    # -------------------------------
    # LOCATION
    # -------------------------------
    lat, lon = get_coordinates(city)
    if lat is None:
        return None

    # -------------------------------
    # WEATHER
    # -------------------------------
    weather = get_weather(lat, lon)
    forecast = get_forecast(lat, lon)

    if "sys" not in weather:
        return None

    sunrise = datetime.fromtimestamp(weather["sys"]["sunrise"]).strftime("%H:%M")
    sunset = datetime.fromtimestamp(weather["sys"]["sunset"]).strftime("%H:%M")

    clouds, humidity = extract_forecast(forecast, obs_date, hour)
    aqi = get_aqi(lat, lon)
    bortle = aqi_to_bortle(aqi)
    moon = get_moon(obs_date)

    # -------------------------------
    # HYBRID OBJECTS
    # -------------------------------
    real = get_real_objects(lat, lon, obs_date, hour)
    stars = get_dataset_objects(obs_date)
    constellations = get_constellation_objects(obs_date)

    all_objs = real + stars + constellations

    if len(all_objs) == 0:
        return None

    df = pd.DataFrame(all_objs)

    # -------------------------------
    # ADD FEATURES
    # -------------------------------
    df["observer_latitude"] = lat
    df["observer_longitude"] = lon
    df["hour"] = hour
    df["month"] = obs_date.month
    df["day"] = obs_date.day
    df["is_night"] = 1
    df["moon_illumination_percent"] = moon
    df["bortle_scale"] = bortle

    # -------------------------------
    # ML PREDICTION
    # -------------------------------
    results = predict(df)
    ranked = rank_objects(results)

    ranked["visibility_score"] = ranked["visibility_score"].apply(
        lambda x: adjust(x, clouds, humidity)
    )

    # -------------------------------
    # FORMAT FOR UI
    # -------------------------------
    ranked = ranked.rename(columns={
        "constellation": "Object",
        "event_type": "Type",
        "direction": "Direction",
        "sky_position": "Position",
        "visibility_probability": "Probability",
        "visibility_score": "Score"
    })

    ranked["Best Time"] = f"{hour}:00"

    best_row = ranked.iloc[0]

    overall_score = float(best_row["Score"] * 100)

    return {
        "lat": round(lat, 2),
        "lon": round(lon, 2),
        "aqi": aqi,
        "bortle": bortle,
        "clouds": clouds,
        "humidity": humidity,
        "moon_illumination": moon,
        "sunrise": sunrise,
        "sunset": sunset,
        "results_df": ranked,
        "best_object": best_row["Object"],
        "best_time": f"{hour}:00",
        "overall_score": overall_score
    }

# -------------------------------
# REAL OBJECTS
# -------------------------------
def get_real_objects(lat, lon, selected_date, hour):
    t = ts.utc(selected_date.year, selected_date.month, selected_date.day, hour)
    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    earth = planets['earth']

    objs = {
        "Mars": planets['mars'],
        "Jupiter": planets['jupiter barycenter'],
        "Venus": planets['venus'],
        "Saturn": planets['saturn barycenter'],
        "Moon": planets['moon']
    }

    data = []

    for name, obj in objs.items():
        astrometric = (earth + observer).at(t).observe(obj)
        alt, az, _ = astrometric.apparent().altaz()

        if alt.degrees > -5:
            data.append({
                "event_type": "planet",
                "constellation": name,
                "elevation_deg": alt.degrees,
                "azimuth_deg": az.degrees,
                "estimated_magnitude": 1.5
            })

    return data


# -------------------------------
# SIMULATED OBJECTS
# -------------------------------
def get_dataset_objects(selected_date):
    season_shift = (selected_date.month - 6) * 5

    stars = ["Sirius", "Vega", "Rigel", "Polaris", "Betelgeuse"]

    data = []

    for s in stars:
        data.append({
            "event_type": "star",
            "constellation": s,
            "elevation_deg": random.uniform(20, 80) + season_shift,
            "azimuth_deg": random.uniform(0, 360),
            "estimated_magnitude": random.uniform(0.5, 3.5)
        })

    return data


def get_constellation_objects(selected_date):
    names = ["Orion", "Ursa Major", "Scorpius", "Cygnus", "Leo"]

    data = []

    for c in names:
        data.append({
            "event_type": "constellation",
            "constellation": c,
            "elevation_deg": random.uniform(30, 70),
            "azimuth_deg": random.uniform(0, 360),
            "estimated_magnitude": random.uniform(1, 4)
        })

    return data


# -------------------------------
# WEATHER
# -------------------------------
def get_coordinates(place):
    url = f"https://api.opencagedata.com/geocode/v1/json?q={place}&key={OPENCAGE_API_KEY}"
    response = requests.get(url).json()

    if "results" in response and len(response["results"]) > 0:
        return response["results"][0]["geometry"]["lat"], response["results"][0]["geometry"]["lng"]
    return None, None


def get_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    return requests.get(url).json()


def get_forecast(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    return requests.get(url).json()


def extract_forecast(forecast, selected_date, hour):
    if "list" not in forecast:
        return 20, 50

    target = selected_date.strftime("%Y-%m-%d")

    for item in forecast["list"]:
        if target in item["dt_txt"] and f"{hour:02d}" in item["dt_txt"]:
            return item["clouds"]["all"], item["main"]["humidity"]

    return 20, 50


def get_aqi(lat, lon):
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
    data = requests.get(url).json()

    if "list" not in data:
        return 3

    return data["list"][0]["main"]["aqi"]


def aqi_to_bortle(aqi):
    return {1: 3, 2: 4, 3: 5, 4: 7, 5: 9}.get(aqi, 5)


def get_moon(selected_date):
    base = datetime(2024, 1, 1)
    days = (selected_date - base.date()).days
    return int(((days % 29.5) / 29.5) * 100)


def adjust(score, clouds, humidity):
    return score * ((100 - clouds)/100) * ((100 - humidity)/100)


# -------------------------------
# BEST TIME (FIXED)
# -------------------------------
def find_best_time(lat, lon, bortle, selected_date, forecast, sunset):
    best_hour = sunset
    best_score = -1

    for h in range(sunset, 24):

        clouds, humidity = extract_forecast(forecast, selected_date, h)

        elevation = max(10, 60 - abs(h - (sunset + 3)) * 5)

        df = pd.DataFrame({
            "observer_latitude": [lat],
            "observer_longitude": [lon],
            "hour": [h],
            "month": [selected_date.month],
            "day": [selected_date.day],
            "is_night": [1],
            "elevation_deg": [elevation],
            "azimuth_deg": [120],
            "estimated_magnitude": [1.5],
            "moon_illumination_percent": [20],
            "bortle_scale": [bortle],
            "event_type": ["planet"],
            "constellation": ["Mars"]
        })

        score = predict(df)["visibility_score"].iloc[0]
        score = adjust(score, clouds, humidity)

        peak_hour = sunset + 3
        darkness_bonus = -((h - peak_hour) ** 2) + 9
        score += max(0, darkness_bonus)

        if score > best_score:
            best_score = score
            best_hour = h

    return best_hour, best_score


# -------------------------------
# PLOT
# -------------------------------
def plot(df):
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    theta = df["azimuth_deg"] * (math.pi / 180)
    r = 90 - df["elevation_deg"]

    ax.scatter(theta, r)

    for i, txt in enumerate(df["constellation"]):
        ax.annotate(txt, (theta.iloc[i], r.iloc[i]))

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════  RESULTS  ════════════════════════════════════════
# ─────────────────────────────────────────────────────────────────────────────

if run_button:
    if not city.strip():
        st.markdown("""
        <div style='background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.3);
                    border-radius:12px;padding:1rem 1.4rem;color:#f87171;
                    font-family:Syne,sans-serif;font-size:0.88rem;'>
            ⚠️  &nbsp; Please enter a city name to begin analysis.
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Loading spinner ────────────────────────────────────────────────────
        with st.spinner(""):
            st.markdown("""
            <style>
            .stSpinner { text-align:center; }
            .stSpinner p {
                font-family:Orbitron,monospace !important;
                font-size:0.78rem !important;
                letter-spacing:0.15em !important;
                color:#22d3ee !important;
                text-transform:uppercase !important;
            }
            </style>
            """, unsafe_allow_html=True)
            # Loading status messages
            status_box = st.empty()
            steps = [
                "🌐  Geocoding location…",
                "🌩️  Fetching weather conditions…",
                "🌫️  Retrieving AQI data…",
                "🌙  Calculating lunar phase…",
                "🔭  Running Skyfield orbital mechanics…",
                "🤖  Running ML classification + regression…",
                "📊  Ranking celestial objects…",
            ]
            import time as _t
            for step in steps:
                status_box.markdown(f"""
                <div style='text-align:center;padding:0.4rem;
                            font-family:Orbitron,monospace;font-size:0.72rem;
                            letter-spacing:0.15em;color:#22d3ee;opacity:0.8;'>
                    {step}
                </div>
                """, unsafe_allow_html=True)
                _t.sleep(0.18)
            status_box.empty()

            result = run_prediction(city, obs_date, obs_time)

        df = result["results_df"]

        # ── Coordinates pill ───────────────────────────────────────────────────
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:0.6rem;margin:0.5rem 0 1.8rem;'>
            <span style='font-family:JetBrains Mono,monospace;font-size:0.78rem;
                         color:#4b6070;'>
                📡 &nbsp;
                <span style='color:#60a5fa;'>{city.title()}</span>
                &nbsp;·&nbsp;
                {result["lat"]}°N, {result["lon"]}°E
                &nbsp;·&nbsp;
                {obs_date.strftime("%d %b %Y")}
                &nbsp;·&nbsp;
                {obs_time.strftime("%H:%M")} local
            </span>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # ENVIRONMENT DASHBOARD
        # ─────────────────────────────────────────────────────────────────────
        section_divider("Atmospheric Conditions")

        aqi_label, aqi_accent = aqi_badge(result["aqi"])
        e1, e2, e3, e4, e5, e6 = st.columns(6)
        with e1:
            metric_pill("AQI", aqi_label, accent=aqi_accent, icon="🌫️")
        with e2:
            metric_pill("Bortle Scale", result["bortle"], "/9", accent="purple", icon="🌑")
        with e3:
            metric_pill("Cloud Cover", result["clouds"], "%",
                        accent="blue" if result["clouds"] < 40 else "gold", icon="☁️")
        with e4:
            metric_pill("Humidity", result["humidity"], "%", accent="cyan", icon="💧")
        with e5:
            metric_pill("Moon", result["moon_illumination"], "%",
                        accent="gold", icon="🌙")
        with e6:
            metric_pill("Sunrise / Sunset",
                        f"{result['sunrise']}  ·  {result['sunset']}", "",
                        accent="gold", icon="🌅")

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # OVERALL INTERPRETATION BANNER
        # ─────────────────────────────────────────────────────────────────────
        vis_label, vis_accent, vis_sym = visibility_alert(result["overall_score"])
        accent_colors = {
            "green": ("#34d399", "rgba(52,211,153,0.08)", "rgba(52,211,153,0.25)"),
            "gold":  ("#fbbf24", "rgba(251,191,36,0.08)", "rgba(251,191,36,0.25)"),
            "red":   ("#f87171", "rgba(248,113,113,0.08)", "rgba(248,113,113,0.25)"),
        }
        c, bg, border = accent_colors[vis_accent]

        st.markdown(f"""
        <div style='background:{bg};border:1px solid {border};border-radius:14px;
                    padding:1.1rem 1.8rem;margin:1.4rem 0;
                    display:flex;align-items:center;justify-content:space-between;
                    flex-wrap:wrap;gap:1rem;'>
            <div style='display:flex;align-items:center;gap:1rem;'>
                <div style='font-size:1.8rem;'>{vis_sym}</div>
                <div>
                    <div style='font-family:Orbitron,monospace;font-size:1rem;
                                font-weight:700;color:{c};letter-spacing:0.08em;'>
                        {vis_label}
                    </div>
                    <div style='font-size:0.8rem;color:#64748b;margin-top:0.2rem;'>
                        Overall visibility score for {city.title()} on {obs_date.strftime("%d %b %Y")}
                    </div>
                </div>
            </div>
            <div style='text-align:right;'>
                <div style='font-family:Orbitron,monospace;font-size:2rem;
                            font-weight:800;color:{c};'>
                    {result["overall_score"]:.1f}
                </div>
                <div style='font-size:0.7rem;letter-spacing:0.15em;
                            text-transform:uppercase;color:#4b6070;'>Score / 100</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # BEST OBJECT RECOMMENDATION CARD
        # ─────────────────────────────────────────────────────────────────────
        best_row = df.iloc[0]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,rgba(251,191,36,0.07),rgba(34,211,238,0.05));
                    border:1px solid rgba(251,191,36,0.28);border-radius:16px;
                    padding:1.4rem 2rem;margin:0.5rem 0 1.8rem;
                    box-shadow:0 0 40px rgba(251,191,36,0.06);'>
            <div style='font-family:Orbitron,monospace;font-size:0.6rem;font-weight:600;
                        letter-spacing:0.25em;text-transform:uppercase;
                        color:#64748b;margin-bottom:0.7rem;'>
                ✦ &nbsp; Tonight's Top Recommendation
            </div>
            <div style='display:flex;align-items:center;
                        justify-content:space-between;flex-wrap:wrap;gap:1rem;'>
                <div style='display:flex;align-items:center;gap:1.2rem;'>
                    <span style='font-size:2.5rem;'>{object_icon(best_row["Type"])}</span>
                    <div>
                        <div style='font-family:Orbitron,monospace;font-size:1.5rem;
                                    font-weight:800;color:#fbbf24;letter-spacing:0.05em;'>
                            {best_row["Object"]}
                        </div>
                        <div style='font-size:0.82rem;color:#94a3b8;margin-top:0.2rem;'>
                            {best_row["Type"]} &nbsp;·&nbsp;
                            Direction: <strong style='color:#e2e8f0;'>{best_row["Direction"]}</strong>
                            &nbsp;·&nbsp;
                            {best_row["Position"]}
                        </div>
                    </div>
                </div>
                <div style='display:flex;gap:1.5rem;text-align:center;'>
                    <div>
                        <div style='font-family:Orbitron,monospace;font-size:1.4rem;
                                    font-weight:700;color:#34d399;'>
                            {best_row["Probability"]*100:.0f}%
                        </div>
                        <div style='font-size:0.65rem;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#4b6070;'>Visibility</div>
                    </div>
                    <div style='width:1px;background:rgba(255,255,255,0.08);'></div>
                    <div>
                        <div style='font-family:Orbitron,monospace;font-size:1.4rem;
                                    font-weight:700;color:#60a5fa;'>
                            {best_row["Score"]:.2f}
                        </div>
                        <div style='font-size:0.65rem;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#4b6070;'>Score</div>
                    </div>
                    <div style='width:1px;background:rgba(255,255,255,0.08);'></div>
                    <div>
                        <div style='font-family:Orbitron,monospace;font-size:1.4rem;
                                    font-weight:700;color:#a78bfa;'>
                            {best_row["Best Time"]}
                        </div>
                        <div style='font-size:0.65rem;letter-spacing:0.1em;
                                    text-transform:uppercase;color:#4b6070;'>Best Time</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ─────────────────────────────────────────────────────────────────────
        # RANKED RESULTS TABLE
        # ─────────────────────────────────────────────────────────────────────
        section_divider("Ranked Celestial Objects")

        # Build display dataframe
        display_df = df.copy()
        display_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(df))])
        display_df["Icon"] = display_df["Type"].apply(object_icon)
        display_df["Object"] = display_df["Icon"] + "  " + display_df["Object"]
        display_df.drop(columns=["Icon"], inplace=True)
        display_df["Probability"] = display_df["Probability"].apply(lambda x: f"{x*100:.1f}%")
        display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            display_df.drop(columns=["Type"]),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank":        st.column_config.TextColumn("Rank",      width="small"),
                "Object":      st.column_config.TextColumn("Object",    width="medium"),
                "Direction":   st.column_config.TextColumn("Direction", width="small"),
                "Position":    st.column_config.TextColumn("Position",  width="medium"),
                "Probability": st.column_config.TextColumn("Visibility %", width="small"),
                "Score":       st.column_config.TextColumn("Score",     width="small"),
                "Best Time":   st.column_config.TextColumn("Best Time", width="small"),
            },
        )

        # ─────────────────────────────────────────────────────────────────────
        # VISUALIZATIONS
        # ─────────────────────────────────────────────────────────────────────
        section_divider("Visual Analysis")

        viz_col1, viz_col2 = st.columns([1.1, 1], gap="large")

        # ── Bar chart ─────────────────────────────────────────────────────────
        with viz_col1:
            st.markdown("""
            <div style='font-family:Orbitron,monospace;font-size:0.65rem;font-weight:600;
                        letter-spacing:0.2em;text-transform:uppercase;color:#4b6070;
                        margin-bottom:0.8rem;'>
                ◈ &nbsp; Visibility Score Distribution
            </div>
            """, unsafe_allow_html=True)

            fig_bar, ax_bar = plt.subplots(figsize=(7, 4.5))
            fig_bar.patch.set_facecolor("#080d14")
            ax_bar.set_facecolor("#080d14")

            objects  = df["Object"].tolist()
            scores   = df["Score"].tolist()
            probs    = df["Probability"].tolist()
            n        = len(objects)
            y_pos    = np.arange(n)

            # Gradient-like colors based on rank
            bar_colors = ["#fbbf24" if i == 0 else
                          f"#{hex(int(96  - i*8))[2:].zfill(2)}"
                          f"{hex(int(165 - i*10))[2:].zfill(2)}"
                          f"{hex(int(250 - i*12))[2:].zfill(2)}"
                          for i in range(n)]

            bars = ax_bar.barh(y_pos, scores, height=0.55,
                               color=bar_colors, zorder=3, alpha=0.88,
                               linewidth=0)

            # Probability overlay dots
            ax2 = ax_bar.twiny()
            ax2.set_xlim(0, 1)
            ax2.scatter([p for p in probs], y_pos,
                        color="#22d3ee", s=55, zorder=5, alpha=0.85,
                        marker="D", linewidths=0)
            ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                                fontfamily="monospace", fontsize=7.5, color="#22d3ee")
            ax2.tick_params(axis="x", colors="#22d3ee", length=3)
            ax2.spines["top"].set_color("rgba(34,211,238,0.25)")
            for spine in ["bottom", "left", "right"]:
                ax2.spines[spine].set_visible(False)

            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(objects, fontfamily="monospace",
                                   fontsize=9, color="#cbd5e1")
            ax_bar.set_xlabel("Visibility Score", fontfamily="monospace",
                              fontsize=8, color="#64748b", labelpad=8)
            ax_bar.tick_params(axis="x", colors="#64748b", labelsize=8, length=3)
            ax_bar.tick_params(axis="y", length=0)
            for spine in ax_bar.spines.values():
                spine.set_color("rgba(96,165,250,0.12)")

            ax_bar.grid(axis="x", color="rgba(96,165,250,0.08)", linestyle="--",
                        linewidth=0.6, zorder=0)
            ax_bar.invert_yaxis()

            # Score labels
            for bar, score in zip(bars, scores):
                ax_bar.text(bar.get_width() + 0.08, bar.get_y() + bar.get_height()/2,
                            f"{score:.2f}", va="center", ha="left",
                            fontfamily="monospace", fontsize=8, color="#94a3b8")

            # Legend
            dot_patch  = mpatches.Patch(color="#22d3ee", label="Visibility %")
            bar_patch  = mpatches.Patch(color="#60a5fa", label="Score")
            ax_bar.legend(handles=[bar_patch, dot_patch],
                          loc="lower right", fontsize=7.5,
                          facecolor="#0d1520", edgecolor="rgba(96,165,250,0.2)",
                          labelcolor="#94a3b8", framealpha=0.8)

            fig_bar.tight_layout(pad=1.2)
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

        # ── Sky Map ───────────────────────────────────────────────────────────
        with viz_col2:
            st.markdown("""
            <div style='font-family:Orbitron,monospace;font-size:0.65rem;font-weight:600;
                        letter-spacing:0.2em;text-transform:uppercase;color:#4b6070;
                        margin-bottom:0.8rem;'>
                ◈ &nbsp; Sky Map (Azimuthal View)
            </div>
            """, unsafe_allow_html=True)

            fig_sky, ax_sky = plt.subplots(figsize=(5, 5),
                                           subplot_kw={"projection": "polar"})
            fig_sky.patch.set_facecolor("#080d14")
            ax_sky.set_facecolor("#020408")

            # Concentric rings
            for r, alpha in [(0.33, 0.15), (0.66, 0.1), (1.0, 0.08)]:
                circle = plt.Circle((0, 0), r, transform=ax_sky.transData._b,
                                    fill=False, color="#60a5fa", alpha=alpha, linewidth=0.7)

            # Direction mapping
            dir_map = {"N": 90, "NE": 45, "E": 0, "SE": -45,
                       "S": -90, "SW": -135, "W": 180, "NW": 135}
            pos_map = {"Near zenith": 0.25, "Zenith": 0.1,
                       "Mid-sky": 0.5, "Low": 0.8, "Horizon": 0.95, "Setting": 0.9}

            for i, row in df.iterrows():
                az_deg  = dir_map.get(row["Direction"], 0)
                az_rad  = math.radians(az_deg)
                elev    = pos_map.get(row["Position"], 0.5)
                prob    = row["Probability"]
                is_best = (i == 0)

                marker_color = "#fbbf24" if is_best else (
                    "#34d399" if prob > 0.75 else
                    "#60a5fa" if prob > 0.5 else "#64748b"
                )
                size = 160 if is_best else int(60 + prob * 80)

                ax_sky.scatter(az_rad, elev, s=size, c=marker_color,
                               zorder=5, alpha=0.9,
                               edgecolors="white" if is_best else "none",
                               linewidths=1.2 if is_best else 0)

                offset = -0.08 if elev < 0.15 else 0.08
                ax_sky.annotate(
                    row["Object"],
                    (az_rad, elev),
                    (az_rad, elev + offset),
                    fontfamily="monospace", fontsize=7,
                    color="#e2e8f0" if is_best else "#94a3b8",
                    fontweight="bold" if is_best else "normal",
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", color=marker_color,
                                   alpha=0.4, lw=0.6) if abs(offset) > 0.05 else None,
                )

            # Compass labels
            ax_sky.set_theta_zero_location("E")
            ax_sky.set_theta_direction(1)
            ax_sky.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4,
                               math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4])
            ax_sky.set_xticklabels(["E", "NE", "N", "NW", "W", "SW", "S", "SE"],
                                   fontfamily="monospace", fontsize=8, color="#60a5fa")
            ax_sky.set_yticks([0, 0.33, 0.66, 1.0])
            ax_sky.set_yticklabels(["Zenith", "60°", "30°", "Horizon"],
                                   fontfamily="monospace", fontsize=7, color="#4b6070")
            ax_sky.set_ylim(0, 1.05)
            ax_sky.tick_params(axis="x", pad=6)

            # Styling
            ax_sky.spines["polar"].set_color("rgba(96,165,250,0.15)")
            ax_sky.yaxis.grid(color="rgba(96,165,250,0.07)", linestyle="--", linewidth=0.5)
            ax_sky.xaxis.grid(color="rgba(96,165,250,0.07)", linestyle="--", linewidth=0.5)

            # Zenith dot
            ax_sky.scatter(0, 0, s=25, c="#22d3ee", zorder=6, alpha=0.6, marker="+")

            fig_sky.tight_layout(pad=0.5)
            st.pyplot(fig_sky, use_container_width=True)
            plt.close(fig_sky)

        # ─────────────────────────────────────────────────────────────────────
        # FOOTER
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("""
        <div style='margin-top:3rem;padding-top:1.5rem;
                    border-top:1px solid rgba(96,165,250,0.1);
                    text-align:center;'>
            <div style='font-family:Orbitron,monospace;font-size:0.58rem;
                        letter-spacing:0.25em;text-transform:uppercase;
                        color:#2d3748;'>
                Hybrid Celestial Visibility Prediction System
                &nbsp;·&nbsp; Skyfield + ML + Weather API
                &nbsp;·&nbsp; For optimal results observe from a dark site
            </div>
        </div>
        """, unsafe_allow_html=True)

else:
    # ── Empty state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;'>
        <div style='font-size:3rem;margin-bottom:1.5rem;opacity:0.25;'>🌌</div>
        <div style='font-family:Orbitron,monospace;font-size:0.7rem;
                    letter-spacing:0.25em;text-transform:uppercase;
                    color:#2d3748;margin-bottom:0.6rem;'>
            Awaiting Coordinates
        </div>
        <div style='font-size:0.82rem;color:#2d3748;max-width:380px;
                    margin:0 auto;line-height:1.8;'>
            Enter your location and observation time above,
            then click <em>Analyze Sky</em> to generate your
            personalized celestial forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)
