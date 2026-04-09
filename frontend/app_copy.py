import sys
import os
import requests
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from datetime import datetime, date

from skyfield.api import load, Topos

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.models.predict import predict
from src.models.ranking import rank_objects


# ─────────────────────────────────────────────────────────────────────────────
# API KEYS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
OPENCAGE_API_KEY = "8910c80cb9c64b73b01e3db45bc27522"
WEATHER_API_KEY  = "ee2e514941fda48295fd461a251082fb"


# ─────────────────────────────────────────────────────────────────────────────
# SKYFIELD  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
ts      = load.timescale()
planets = load('de421.bsp')


# ─────────────────────────────────────────────────────────────────────────────
# REAL OBJECTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def get_real_objects(lat, lon, selected_date, hour):
    t        = ts.utc(selected_date.year, selected_date.month, selected_date.day, hour)
    observer = Topos(latitude_degrees=lat, longitude_degrees=lon)
    earth    = planets['earth']

    objs = {
        "Mars":    planets['mars'],
        "Jupiter": planets['jupiter barycenter'],
        "Venus":   planets['venus'],
        "Saturn":  planets['saturn barycenter'],
        "Moon":    planets['moon'],
    }

    data = []
    for name, obj in objs.items():
        astrometric = (earth + observer).at(t).observe(obj)
        alt, az, _  = astrometric.apparent().altaz()
        if alt.degrees > -5:
            data.append({
                "event_type":          "planet",
                "constellation":       name,
                "elevation_deg":       alt.degrees,
                "azimuth_deg":         az.degrees,
                "estimated_magnitude": 1.5,
            })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED OBJECTS  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def get_dataset_objects(selected_date):
    season_shift = (selected_date.month - 6) * 5
    stars = ["Sirius", "Vega", "Rigel", "Polaris", "Betelgeuse"]
    data  = []
    for s in stars:
        data.append({
            "event_type":          "star",
            "constellation":       s,
            "elevation_deg":       random.uniform(20, 80) + season_shift,
            "azimuth_deg":         random.uniform(0, 360),
            "estimated_magnitude": random.uniform(0.5, 3.5),
        })
    return data


def get_constellation_objects(selected_date):
    names = ["Orion", "Ursa Major", "Scorpius", "Cygnus", "Leo"]
    data  = []
    for c in names:
        data.append({
            "event_type":          "constellation",
            "constellation":       c,
            "elevation_deg":       random.uniform(30, 70),
            "azimuth_deg":         random.uniform(0, 360),
            "estimated_magnitude": random.uniform(1, 4),
        })
    return data


# ─────────────────────────────────────────────────────────────────────────────
# WEATHER / GEO  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def get_coordinates(place):
    url      = f"https://api.opencagedata.com/geocode/v1/json?q={place}&key={OPENCAGE_API_KEY}"
    response = requests.get(url).json()
    if "results" in response and len(response["results"]) > 0:
        return (response["results"][0]["geometry"]["lat"],
                response["results"][0]["geometry"]["lng"])
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
    url  = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}"
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
    return score * ((100 - clouds) / 100) * ((100 - humidity) / 100)


# ─────────────────────────────────────────────────────────────────────────────
# BEST TIME  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────
def find_best_time(lat, lon, bortle, selected_date, forecast, sunset):
    best_hour  = sunset
    best_score = -1

    for h in range(sunset, 24):
        clouds, humidity = extract_forecast(forecast, selected_date, h)
        elevation        = max(10, 60 - abs(h - (sunset + 3)) * 5)

        df_tmp = pd.DataFrame({
            "observer_latitude":      [lat],
            "observer_longitude":     [lon],
            "hour":                   [h],
            "month":                  [selected_date.month],
            "day":                    [selected_date.day],
            "is_night":               [1],
            "elevation_deg":          [elevation],
            "azimuth_deg":            [120],
            "estimated_magnitude":    [1.5],
            "moon_illumination_percent": [20],
            "bortle_scale":           [bortle],
            "event_type":             ["planet"],
            "constellation":          ["Mars"],
        })

        score      = predict(df_tmp)["visibility_score"].iloc[0]
        score      = adjust(score, clouds, humidity)
        peak_hour  = sunset + 3
        darkness_bonus = -((h - peak_hour) ** 2) + 9
        score     += max(0, darkness_bonus)

        if score > best_score:
            best_score = score
            best_hour  = h

    return best_hour, best_score


# ─────────────────────────────────────────────────────────────────────────────
# PLOT  (original logic kept; styling enhanced)
# ─────────────────────────────────────────────────────────────────────────────
def plot(df):
    """
    Original sky-map logic preserved.
    Visual styling updated to Apple light theme.
    """
    fig = plt.figure(figsize=(5.8, 5.8))
    ax  = fig.add_subplot(111, polar=True)

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f5f5f7")

    theta = df["azimuth_deg"] * (math.pi / 180)
    r     = 90 - df["elevation_deg"]

    # Color by event type — Apple palette
    color_map = {"planet": "#0071e3", "star": "#ff9500", "constellation": "#5856d6"}
    colors    = df["event_type"].map(color_map).fillna("#34c759")
    sizes     = [140 if i == 0 else 60 for i in range(len(df))]

    ax.scatter(theta, r, c=colors, s=sizes, zorder=5, alpha=0.90,
               edgecolors="white", linewidths=0.6)

    for i, txt in enumerate(df["constellation"]):
        ax.annotate(
            txt,
            (theta.iloc[i], r.iloc[i]),
            xytext=(theta.iloc[i], r.iloc[i] - 5),
            fontfamily="monospace",
            fontsize=7,
            color="#1d1d1f" if i == 0 else "#6e6e73",
            fontweight="bold" if i == 0 else "normal",
            ha="center",
        )

    # Original orientation (unchanged)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4,
                   math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4])
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       fontfamily="monospace", fontsize=8, color="#6e6e73")
    ax.set_yticklabels([])

    ax.spines["polar"].set_color((0, 0, 0, 0.10))
    ax.yaxis.grid(color=(0, 0, 0, 0.06), linestyle="--", linewidth=0.5)
    ax.xaxis.grid(color=(0, 0, 0, 0.06), linestyle="--", linewidth=0.5)

    # Legend
    handles = [
        mpatches.Patch(color="#0071e3", label="Planet"),
        mpatches.Patch(color="#ff9500", label="Star"),
        mpatches.Patch(color="#5856d6", label="Constellation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor="#ffffff", edgecolor=(0, 0, 0, 0.1),
              labelcolor="#1d1d1f", framealpha=0.95,
              bbox_to_anchor=(1.28, -0.05))

    fig.tight_layout(pad=0.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#
#                         UI  —  REDESIGNED FRONTEND
#
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Celestial Vision | AI Sky Predictor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS  — Apple-inspired: light, airy, premium typography
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg-base:       #f5f5f7;
    --bg-white:      #ffffff;
    --bg-card:       #ffffff;
    --bg-glass:      rgba(255,255,255,0.82);
    --border:        rgba(0,0,0,0.08);
    --border-med:    rgba(0,0,0,0.14);
    --accent-navy:   #1d1d1f;
    --accent-blue:   #0071e3;
    --accent-indigo: #5856d6;
    --accent-teal:   #32ade6;
    --accent-gold:   #bf8700;
    --accent-green:  #34c759;
    --accent-red:    #ff3b30;
    --accent-orange: #ff9500;
    --text-primary:  #1d1d1f;
    --text-secondary:#6e6e73;
    --text-tertiary: #aeaeb2;
    --shadow-soft:   0 2px 20px rgba(0,0,0,0.06);
    --shadow-med:    0 4px 40px rgba(0,0,0,0.10);
    --radius:        16px;
    --radius-sm:     10px;
}

.stApp {
    background: var(--bg-base);
    font-family: 'DM Sans', -apple-system, sans-serif;
    color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3.5rem 5rem; max-width: 1300px; }
.stDeployButton { display: none; }

h1,h2,h3 { font-family: 'Playfair Display', Georgia, serif; }

/* Inputs */
.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stTimeInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-white) !important;
    border: 1px solid var(--border-med) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    box-shadow: var(--shadow-soft) !important;
}
.stTextInput > div > div > input:focus,
.stDateInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(0,113,227,0.15) !important;
    outline: none !important;
}
.stSlider > div > div > div > div {
    background: var(--accent-blue) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
}

/* Labels */
label, .stTextInput label, .stDateInput label,
.stSlider label, .stSelectbox label {
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* Button */
.stButton > button {
    background: var(--accent-blue) !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    color: #ffffff !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.18s cubic-bezier(0.25,0.46,0.45,0.94) !important;
    width: 100% !important;
    box-shadow: 0 2px 12px rgba(0,113,227,0.30) !important;
}
.stButton > button:hover {
    background: #0077ed !important;
    box-shadow: 0 4px 20px rgba(0,113,227,0.40) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Dataframe */
.stDataFrame thead th {
    background: rgba(0,0,0,0.03) !important;
    color: var(--text-secondary) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; border: 1px solid var(--border) !important; box-shadow: var(--shadow-soft); }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-blue) !important; }

/* Alert boxes */
.stAlert { border-radius: 12px !important; font-family: 'DM Sans', sans-serif !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section_label(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1.2rem;margin:2.4rem 0 1.2rem;">
        <span style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;
                     letter-spacing:0.14em;text-transform:uppercase;color:var(--text-tertiary,#aeaeb2);
                     white-space:nowrap;">{text}</span>
        <div style="flex:1;height:1px;background:rgba(0,0,0,0.08);"></div>
    </div>
    """, unsafe_allow_html=True)


def metric_pill(label, value, unit="", accent="blue", icon=""):
    color_map = {
        "blue":   ("#0071e3", "rgba(0,113,227,0.07)"),
        "cyan":   ("#32ade6", "rgba(50,173,230,0.07)"),
        "gold":   ("#bf8700", "rgba(191,135,0,0.07)"),
        "green":  ("#34c759", "rgba(52,199,89,0.07)"),
        "red":    ("#ff3b30", "rgba(255,59,48,0.07)"),
        "purple": ("#5856d6", "rgba(88,86,214,0.07)"),
    }
    c, bg = color_map.get(accent, color_map["blue"])
    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.08);border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;
                box-shadow:0 2px 16px rgba(0,0,0,0.06);">
        <div style="font-size:1.3rem;margin-bottom:0.3rem;">{icon}</div>
        <div style="font-family:Playfair Display,serif;font-size:1.15rem;font-weight:700;
                    color:{c};line-height:1.1;">
            {value}<span style="font-family:DM Sans,sans-serif;font-size:0.65rem;
                                 font-weight:500;opacity:0.65;margin-left:2px;">{unit}</span>
        </div>
        <div style="font-family:DM Sans,sans-serif;font-size:0.62rem;font-weight:500;
                    letter-spacing:0.08em;text-transform:uppercase;
                    color:#6e6e73;margin-top:0.35rem;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def object_icon(t):
    t = str(t).lower()
    if "planet" in t:       return "🌍"
    if "star" in t:         return "⭐"
    if "constell" in t:     return "🌌"
    if "moon" in t:         return "🌕"
    return "✦"


AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_ACCENT = {1: "green", 2: "green", 3: "gold", 4: "red", 5: "red"}


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:3rem 0 1.5rem;text-align:center;">
    <div style="display:inline-flex;align-items:center;justify-content:center;
                width:56px;height:56px;border-radius:14px;
                background:linear-gradient(145deg,#1d1d1f,#3a3a3c);
                box-shadow:0 4px 20px rgba(0,0,0,0.18);
                font-size:1.5rem;margin-bottom:1.4rem;">🔭</div>

    <h1 style="font-family:Georgia,serif;
               font-size:clamp(1.8rem,4vw,3rem);font-weight:700;
               letter-spacing:-0.02em;color:#1d1d1f;
               margin:0 0 0.5rem;line-height:1.1;">
        Celestial Visibility
    </h1>
    <div style="font-size:clamp(0.8rem,1.8vw,1rem);
                font-weight:400;letter-spacing:0.01em;color:#6e6e73;margin-bottom:1rem;">
        AI-Powered Sky Prediction System
    </div>
    <p style="font-size:0.88rem;color:#aeaeb2;
              max-width:480px;margin:0 auto 2rem;line-height:1.8;font-weight:400;">
        Real-time weather &amp; atmospheric analysis combined with
        orbital mechanics to rank your best celestial targets tonight.
    </p>
    <div style="width:40px;height:2px;background:#0071e3;border-radius:2px;margin:0 auto;"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
section_label("Observation Parameters")

st.markdown("""
<div style="background:#ffffff;border:1px solid rgba(0,0,0,0.08);border-radius:18px;
            padding:1.6rem 2rem 0.8rem;margin-bottom:0.5rem;
            box-shadow:0 2px 24px rgba(0,0,0,0.07);">
    <div style="font-family:DM Sans,sans-serif;font-size:0.68rem;font-weight:600;
                letter-spacing:0.12em;text-transform:uppercase;color:#aeaeb2;
                margin-bottom:1.1rem;">◈ &nbsp; Location &amp; Time</div>
</div>
""", unsafe_allow_html=True)

col_loc, col_date, col_hour, col_btn = st.columns([3, 2, 2, 1.5])

with col_loc:
    place = st.text_input(
        "📍 City / Location",
        value="Pune",
        placeholder="e.g. Pune, Mumbai, New York…",
        help="Enter any city — coordinates are resolved automatically via OpenCage geocoding.",
    )

with col_date:
    selected_date = st.date_input(
        "📅 Observation Date",
        value=date.today(),
        help="The calendar date of your observation session.",
    )

with col_hour:
    hour = st.slider(
        "🕐 Hour (24 h)",
        min_value=0,
        max_value=23,
        value=21,
        help="Local hour of observation. Night hours (after sunset) give best results.",
    )

with col_btn:
    st.markdown('<div style="height:1.85rem"></div>', unsafe_allow_html=True)
    run_button = st.button("⚡ Analyze Sky", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_button:

    if not place.strip():
        st.markdown("""
        <div style="background:#fff2f2;border:1px solid rgba(255,59,48,0.2);
                    border-radius:12px;padding:1rem 1.4rem;color:#ff3b30;
                    font-family:DM Sans,sans-serif;font-size:0.88rem;
                    box-shadow:0 2px 12px rgba(255,59,48,0.08);">
            ⚠️ &nbsp; Please enter a city name to begin analysis.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Loading ──────────────────────────────────────────────────────────────
    steps = [
        "🌐  Geocoding location coordinates…",
        "🌩️  Fetching real-time weather data…",
        "🌫️  Retrieving air quality index…",
        "🌙  Computing lunar phase…",
        "🔭  Running Skyfield orbital mechanics…",
        "🤖  Running ML classification + regression…",
        "📊  Ranking celestial objects…",
    ]

    status_slot = st.empty()
    prog_slot   = st.empty()

    with st.spinner(""):
        for i, step in enumerate(steps):
            status_slot.markdown(f"""
            <div style="text-align:center;padding:0.5rem;font-family:DM Sans,sans-serif;
                        font-size:0.82rem;letter-spacing:0.02em;color:#6e6e73;">
                {step}
            </div>
            """, unsafe_allow_html=True)
            prog_slot.progress((i + 1) / len(steps))
            import time as _t; _t.sleep(0.12)

        # ── GEOCODING ─────────────────────────────────────────────────────────
        lat, lon = get_coordinates(place)

    status_slot.empty()
    prog_slot.empty()

    if lat is None:
        st.markdown("""
        <div style="background:#fff2f2;border:1px solid rgba(255,59,48,0.2);
                    border-radius:12px;padding:1rem 1.4rem;color:#ff3b30;
                    font-family:DM Sans,sans-serif;box-shadow:0 2px 12px rgba(255,59,48,0.08);">
            ❌ &nbsp; Could not resolve that location. Check the city name and try again.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── WEATHER ───────────────────────────────────────────────────────────────
    weather  = get_weather(lat, lon)
    forecast = get_forecast(lat, lon)

    if "sys" not in weather:
        st.markdown("""
        <div style="background:#fff2f2;border:1px solid rgba(255,59,48,0.2);
                    border-radius:12px;padding:1rem 1.4rem;color:#ff3b30;
                    font-family:DM Sans,sans-serif;">
            ❌ &nbsp; Weather API failed. Check your API key or try again later.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    sunrise = datetime.fromtimestamp(weather["sys"]["sunrise"]).hour
    sunset  = datetime.fromtimestamp(weather["sys"]["sunset"]).hour

    # ── ENV DATA ──────────────────────────────────────────────────────────────
    clouds, humidity = extract_forecast(forecast, selected_date, hour)
    aqi              = get_aqi(lat, lon)
    bortle           = aqi_to_bortle(aqi)
    moon             = get_moon(selected_date)

    # ── Coordinates banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin:1rem 0 0.5rem;font-family:"DM Mono',"DM Sans",monospace;
                font-size:0.78rem;color:#aeaeb2;">
        📡 &nbsp;
        <span style="color:#1d1d1f;font-weight:500;">{place.title()}</span>
        &nbsp;·&nbsp; {lat:.4f}°N, {lon:.4f}°E
        &nbsp;·&nbsp; {selected_date.strftime("%d %b %Y")}
        &nbsp;·&nbsp; {hour:02d}:00 local
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ENVIRONMENT DASHBOARD
    # ─────────────────────────────────────────────────────────────────────────
    section_label("Atmospheric Conditions")

    e1, e2, e3, e4, e5, e6 = st.columns(6)

    with e1:
        metric_pill("Air Quality", AQI_LABELS.get(aqi, "—"),
                    accent=AQI_ACCENT.get(aqi, "blue"), icon="🌫️")
    with e2:
        metric_pill("Bortle Scale", f"{bortle}", "/9", accent="purple", icon="🌑")
    with e3:
        metric_pill("Cloud Cover", f"{clouds}", "%",
                    accent="blue" if clouds < 40 else "gold", icon="☁️")
    with e4:
        metric_pill("Humidity", f"{humidity}", "%", accent="cyan", icon="💧")
    with e5:
        metric_pill("Moon", f"{moon}", "%",
                    accent="gold" if moon > 50 else "blue", icon="🌙")
    with e6:
        metric_pill("Sunrise / Sunset",
                    f"{sunrise:02d}:00 – {sunset:02d}:00",
                    accent="gold", icon="🌅")

    st.markdown('<div style="height:0.6rem"></div>', unsafe_allow_html=True)

    # ── Daytime guard ─────────────────────────────────────────────────────────
    if not (hour >= sunset or hour <= sunrise):
        st.markdown(f"""
        <div style="background:#fffbe6;border:1px solid rgba(191,135,0,0.22);
                    border-radius:12px;padding:1.1rem 1.6rem;color:#bf8700;
                    font-family:DM Sans,sans-serif;font-size:0.88rem;
                    box-shadow:0 2px 12px rgba(191,135,0,0.06);">
            ☀️ &nbsp; Selected hour <strong>{hour:02d}:00</strong> is during daytime
            (Sunrise {sunrise:02d}:00 → Sunset {sunset:02d}:00).
            Celestial objects are not visible. Please choose a night-time hour.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECT GENERATION + ML PIPELINE  (backend — untouched)
    # ─────────────────────────────────────────────────────────────────────────
    real           = get_real_objects(lat, lon, selected_date, hour)
    stars          = get_dataset_objects(selected_date)
    constellations = get_constellation_objects(selected_date)
    all_objs       = real + stars + constellations

    df = pd.DataFrame(all_objs)
    df["observer_latitude"]       = lat
    df["observer_longitude"]      = lon
    df["hour"]                    = hour
    df["month"]                   = selected_date.month
    df["day"]                     = selected_date.day
    df["is_night"]                = 1
    df["moon_illumination_percent"] = moon
    df["bortle_scale"]            = bortle

    results = predict(df)
    ranked  = rank_objects(results)

    ranked["visibility_score"] = ranked["visibility_score"].apply(
        lambda x: adjust(x, clouds, humidity)
    )

    display_df = ranked.copy()
    display_df["visibility_probability"] = (
        display_df["visibility_probability"] * 100
    ).round(2)

    # ─────────────────────────────────────────────────────────────────────────
    # INTERPRETATION BANNER
    # ─────────────────────────────────────────────────────────────────────────
    best_score = ranked.iloc[0]["visibility_score"]

    if best_score > 0.8:
        vis_label, vis_accent_c, vis_bg, vis_border, vis_sym = (
            "Excellent Viewing Conditions", "#34c759",
            "#f2fff6", "rgba(52,199,89,0.22)", "✦")
    elif best_score > 0.5:
        vis_label, vis_accent_c, vis_bg, vis_border, vis_sym = (
            "Moderate Visibility", "#bf8700",
            "#fffbe6", "rgba(191,135,0,0.22)", "◈")
    else:
        vis_label, vis_accent_c, vis_bg, vis_border, vis_sym = (
            "Poor Visibility Tonight", "#ff3b30",
            "#fff2f2", "rgba(255,59,48,0.22)", "✕")

    score_pct = min(100, best_score * 100)

    st.markdown(f"""
    <div style="background:{vis_bg};border:1px solid {vis_border};border-radius:16px;
                padding:1.3rem 1.8rem;margin:1rem 0 1.5rem;
                box-shadow:0 2px 18px rgba(0,0,0,0.06);
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div style="display:flex;align-items:center;gap:1rem;">
            <span style="font-size:1.5rem;">{vis_sym}</span>
            <div>
                <div style="font-family:Playfair Display,serif;font-size:1rem;
                            font-weight:700;color:{vis_accent_c};letter-spacing:0;">
                    {vis_label}
                </div>
                <div style="font-family:DM Sans,sans-serif;font-size:0.78rem;
                            color:#6e6e73;margin-top:0.2rem;">
                    {place.title()} · {selected_date.strftime("%d %b %Y")} · {hour:02d}:00
                </div>
            </div>
        </div>
        <div style="text-align:right;">
            <div style="font-family:Playfair Display,serif;font-size:2rem;
                        font-weight:700;color:{vis_accent_c};">{score_pct:.1f}</div>
            <div style="font-family:DM Sans,sans-serif;font-size:0.62rem;
                        letter-spacing:0.12em;text-transform:uppercase;color:#aeaeb2;">Score / 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TOP OBJECT RECOMMENDATION CARD
    # ─────────────────────────────────────────────────────────────────────────
    top = ranked.iloc[0]
    top_prob  = display_df.iloc[0]["visibility_probability"]
    top_icon  = object_icon(top.get("event_type", ""))
    top_dir   = top.get("direction",    "—")
    top_pos   = top.get("sky_position", "—")
    top_name  = top.get("constellation", "—")

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.09);border-radius:18px;
                padding:1.5rem 1.8rem;margin-bottom:1.5rem;
                box-shadow:0 4px 32px rgba(0,0,0,0.08);">
        <div style="font-family:DM Sans,sans-serif;font-size:0.65rem;font-weight:600;
                    letter-spacing:0.14em;text-transform:uppercase;
                    color:#aeaeb2;margin-bottom:1rem;">
            ✦ &nbsp; Tonight's Top Recommendation
        </div>
        <div style="display:flex;align-items:center;
                    justify-content:space-between;flex-wrap:wrap;gap:1rem;">
            <div style="display:flex;align-items:center;gap:1.2rem;">
                <span style="font-size:2.2rem;">{top_icon}</span>
                <div>
                    <div style="font-family:Playfair Display,serif;font-size:1.5rem;
                                font-weight:700;color:#1d1d1f;letter-spacing:0;">
                        {top_name}
                    </div>
                    <div style="font-family:DM Sans,sans-serif;font-size:0.78rem;
                                color:#6e6e73;margin-top:0.15rem;">
                        {str(top.get("event_type","")).title()}
                        &nbsp;·&nbsp; Direction:
                        <strong style="color:#1d1d1f;">{top_dir}</strong>
                        &nbsp;·&nbsp; {top_pos}
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:2rem;text-align:center;align-items:center;">
                <div>
                    <div style="font-family:Playfair Display,serif;font-size:1.5rem;
                                font-weight:700;color:#34c759;">{top_prob:.1f}%</div>
                    <div style="font-family:DM Sans,sans-serif;font-size:0.6rem;
                                letter-spacing:0.1em;text-transform:uppercase;color:#aeaeb2;">Visibility</div>
                </div>
                <div style="width:1px;height:36px;background:rgba(0,0,0,0.08);"></div>
                <div>
                    <div style="font-family:Playfair Display,serif;font-size:1.5rem;
                                font-weight:700;color:#0071e3;">
                        {top["visibility_score"]:.2f}
                    </div>
                    <div style="font-family:DM Sans,sans-serif;font-size:0.6rem;
                                letter-spacing:0.1em;text-transform:uppercase;color:#aeaeb2;">Score</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS TABLE
    # ─────────────────────────────────────────────────────────────────────────
    section_label("Ranked Celestial Objects")

    table_df = display_df.copy()
    table_df.insert(0, "Rank", [f"#{i+1}" for i in range(len(table_df))])
    table_df["Object"] = table_df["event_type"].apply(object_icon) + "  " + table_df["constellation"]
    table_df["Prob %"] = table_df["visibility_probability"].apply(lambda x: f"{x:.1f}%")
    table_df["Score"]  = table_df["visibility_score"].apply(lambda x: f"{x:.2f}")

    st.dataframe(
        table_df[["Rank", "Object", "direction", "sky_position", "Prob %", "Score"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank":        st.column_config.TextColumn("Rank",      width="small"),
            "Object":      st.column_config.TextColumn("Object",    width="medium"),
            "direction":   st.column_config.TextColumn("Direction", width="small"),
            "sky_position":st.column_config.TextColumn("Position",  width="medium"),
            "Prob %":      st.column_config.TextColumn("Visibility %", width="small"),
            "Score":       st.column_config.TextColumn("Score",     width="small"),
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # BEST VIEWING TIME
    # ─────────────────────────────────────────────────────────────────────────
    section_label("Optimal Viewing Window")

    best_hour, _ = find_best_time(lat, lon, bortle, selected_date, forecast, sunset)

    st.markdown(f"""
    <div style="background:#ffffff;border:1px solid rgba(0,0,0,0.09);border-radius:16px;
                padding:1.4rem 2rem;margin-bottom:0.5rem;
                box-shadow:0 2px 20px rgba(0,0,0,0.07);
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div>
            <div style="font-family:DM Sans,sans-serif;font-size:0.65rem;font-weight:600;
                        letter-spacing:0.14em;text-transform:uppercase;color:#aeaeb2;
                        margin-bottom:0.5rem;">
                🕒 &nbsp; ML-Recommended Best Time
            </div>
            <div style="font-family:Playfair Display,serif;font-size:2.4rem;
                        font-weight:700;color:#0071e3;line-height:1;">
                {best_hour:02d}:00
            </div>
            <div style="font-family:DM Sans,sans-serif;font-size:0.78rem;
                        color:#6e6e73;margin-top:0.4rem;">
                Optimal darkness + minimal atmospheric interference
            </div>
        </div>
        <div style="text-align:right;font-family:"DM Mono',"DM Sans",monospace;
                    font-size:0.78rem;color:#6e6e73;line-height:2;">
            Sunset &nbsp; {sunset:02d}:00<br>
            Sunrise &nbsp; {sunrise:02d}:00<br>
            Bortle &nbsp; {bortle} / 9
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATIONS
    # ─────────────────────────────────────────────────────────────────────────
    section_label("Visual Analysis")

    viz_l, viz_r = st.columns([1.1, 1], gap="large")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    with viz_l:
        st.markdown("""
        <div style="font-family:DM Sans,sans-serif;font-size:0.65rem;font-weight:600;
                    letter-spacing:0.12em;text-transform:uppercase;color:#aeaeb2;
                    margin-bottom:0.8rem;">◈ &nbsp; Visibility Score Distribution</div>
        """, unsafe_allow_html=True)

        fig_bar, ax_bar = plt.subplots(figsize=(7, max(4, len(ranked) * 0.45 + 1)))
        fig_bar.patch.set_facecolor("#ffffff")
        ax_bar.set_facecolor("#ffffff")

        objects_list = ranked["constellation"].tolist()
        scores_list  = ranked["visibility_score"].tolist()
        probs_list   = display_df["visibility_probability"].tolist()
        n            = len(objects_list)
        y_pos        = np.arange(n)

        # Color: blue for #1, muted blues for rest
        bar_colors = []
        for i in range(n):
            if i == 0:
                bar_colors.append("#0071e3")
            else:
                frac = 1 - (i / max(n - 1, 1)) * 0.45
                bar_colors.append(
                    f"#{int(0*frac+180*(1-frac)):02x}{int(113*frac+180*(1-frac)):02x}{int(227*frac+200*(1-frac)):02x}"
                )

        bars = ax_bar.barh(y_pos, scores_list, height=0.52,
                           color=bar_colors, zorder=3, alpha=0.90)

        # Probability dots on secondary axis
        ax2 = ax_bar.twiny()
        ax2.set_xlim(0, 100)
        ax2.scatter(probs_list, y_pos, color="#5856d6", s=52,
                    zorder=5, alpha=0.80, marker="D", linewidths=0)
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                            fontfamily="monospace", fontsize=7.5, color="#5856d6")
        ax2.tick_params(axis="x", colors="#5856d6", length=3)
        ax2.spines["top"].set_color((88/255, 86/255, 214/255, 0.2))
        for sp in ["bottom", "left", "right"]:
            ax2.spines[sp].set_visible(False)

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(objects_list, fontfamily="monospace",
                               fontsize=9, color="#1d1d1f")
        ax_bar.set_xlabel("Visibility Score", fontfamily="monospace",
                          fontsize=8, color="#6e6e73", labelpad=8)
        ax_bar.tick_params(axis="x", colors="#6e6e73", labelsize=8, length=3)
        ax_bar.tick_params(axis="y", length=0)
        for sp in ax_bar.spines.values():
            sp.set_color((0, 0, 0, 0.08))
        ax_bar.grid(axis="x", color=(0, 0, 0, 0.05),
                    linestyle="--", linewidth=0.6, zorder=0)
        ax_bar.invert_yaxis()

        for bar, score in zip(bars, scores_list):
            ax_bar.text(bar.get_width() + 0.005,
                        bar.get_y() + bar.get_height() / 2,
                        f"{score:.2f}", va="center", ha="left",
                        fontfamily="monospace", fontsize=7.5, color="#6e6e73")

        h_bars = mpatches.Patch(color="#0071e3", label="Visibility Score")
        h_dots = mpatches.Patch(color="#5856d6", label="Probability %")
        ax_bar.legend(handles=[h_bars, h_dots], loc="lower right", fontsize=7.5,
                      facecolor="#ffffff", edgecolor=(0, 0, 0, 0.1),
                      labelcolor="#1d1d1f", framealpha=0.95)

        fig_bar.tight_layout(pad=1.2)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # ── Sky map  (original plot() function — logic unchanged) ─────────────────
    with viz_r:
        st.markdown("""
        <div style="font-family:DM Sans,sans-serif;font-size:0.65rem;font-weight:600;
                    letter-spacing:0.12em;text-transform:uppercase;color:#aeaeb2;
                    margin-bottom:0.8rem;">◈ &nbsp; Sky Map (Azimuthal Projection)</div>
        """, unsafe_allow_html=True)
        st.pyplot(plot(ranked), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:3.5rem;padding-top:1.5rem;
                border-top:1px solid rgba(0,0,0,0.07);text-align:center;">
        <div style="font-family:DM Sans,sans-serif;font-size:0.72rem;
                    letter-spacing:0.06em;color:#aeaeb2;">
            Celestial Visibility System
            &nbsp;·&nbsp; Skyfield + ML + OpenWeather + OpenCage
            &nbsp;·&nbsp; For best results observe from a dark-sky site
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE  (shown before first run)
# ─────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("""
    <div style="text-align:center;padding:5rem 2rem;">
        <div style="font-size:3rem;opacity:0.18;margin-bottom:1.5rem;">🌌</div>
        <div style="font-family:Playfair Display,serif;font-size:1.3rem;
                    font-weight:600;color:#1d1d1f;margin-bottom:0.6rem;">
            Ready for Observation
        </div>
        <div style="font-family:DM Sans,sans-serif;font-size:0.88rem;
                    color:#6e6e73;max-width:360px;margin:0 auto;line-height:1.85;">
            Enter your city and observation hour above,<br>
            then click <em>Analyze Sky</em> to generate<br>
            your personalized celestial forecast.
        </div>
    </div>
    """, unsafe_allow_html=True)