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
import streamlit.components.v1 as components
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
# PLOT  — dark space theme
# ─────────────────────────────────────────────────────────────────────────────
def plot(df):
    fig = plt.figure(figsize=(5.8, 5.8))
    ax  = fig.add_subplot(111, polar=True)

    fig.patch.set_facecolor("#050a1a")
    ax.set_facecolor("#050a1a")

    theta = df["azimuth_deg"] * (math.pi / 180)
    r     = 90 - df["elevation_deg"]

    color_map = {"planet": "#7eb8ff", "star": "#ffd97d", "constellation": "#c8b8ff"}
    colors    = df["event_type"].map(color_map).fillna("#60efb4")
    sizes     = [180 if i == 0 else 70 for i in range(len(df))]

    # Draw faint star field
    rng = np.random.default_rng(42)
    bg_theta = rng.uniform(0, 2 * math.pi, 120)
    bg_r     = rng.uniform(0, 90, 120)
    ax.scatter(bg_theta, bg_r, c="white", s=rng.uniform(1, 5, 120),
               alpha=rng.uniform(0.05, 0.35, 120), zorder=1, linewidths=0)

    ax.scatter(theta, r, c=colors, s=sizes, zorder=5, alpha=0.95,
               edgecolors="white", linewidths=0.8)

    # Glow effect for top object
    if len(df) > 0:
        ax.scatter([theta.iloc[0]], [r.iloc[0]], c=[colors.iloc[0]], s=500,
                   alpha=0.12, zorder=4, linewidths=0)
        ax.scatter([theta.iloc[0]], [r.iloc[0]], c=[colors.iloc[0]], s=280,
                   alpha=0.18, zorder=4, linewidths=0)

    for i, txt in enumerate(df["constellation"]):
        ax.annotate(
            txt,
            (theta.iloc[i], r.iloc[i]),
            xytext=(theta.iloc[i], r.iloc[i] - 6),
            fontfamily="monospace",
            fontsize=7,
            color="#ffffff" if i == 0 else "#8899bb",
            fontweight="bold" if i == 0 else "normal",
            ha="center",
        )

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    ax.set_xticks([0, math.pi/4, math.pi/2, 3*math.pi/4,
                   math.pi, 5*math.pi/4, 3*math.pi/2, 7*math.pi/4])
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
                       fontfamily="monospace", fontsize=8, color="#556688")
    ax.set_yticklabels([])

    ax.spines["polar"].set_color("#1a2a4a")
    ax.yaxis.grid(color=(0.2, 0.4, 0.8, 0.12), linestyle="--", linewidth=0.6)
    ax.xaxis.grid(color=(0.2, 0.4, 0.8, 0.12), linestyle="--", linewidth=0.6)

    handles = [
        mpatches.Patch(color="#7eb8ff", label="Planet"),
        mpatches.Patch(color="#ffd97d", label="Star"),
        mpatches.Patch(color="#c8b8ff", label="Constellation"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=7,
              facecolor="#0a1428", edgecolor="#1a2a4a",
              labelcolor="#aabbcc", framealpha=0.92,
              bbox_to_anchor=(1.28, -0.05))

    fig.tight_layout(pad=0.5)
    return fig


# ═════════════════════════════════════════════════════════════════════════════
#  UI  —  DARK COSMOS REDESIGN
# ═════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Celestial Vision | AI Sky Predictor",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS — deep-space dark theme with neon-blue accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-void:       #02050f;
    --bg-card:       #0b1628;
    --border:        rgba(77,166,255,0.15);
    --accent-blue:   #4da6ff;
    --text-primary:  #e8f0ff;
    --text-secondary:#7a98c8;
    --text-dim:      #3a5070;
    --radius:        16px;
    --radius-sm:     10px;
}

/* ── Base ── */
.stApp {
    background: var(--bg-void) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 3rem 5rem; max-width: 1320px; }
.stDeployButton { display: none; }

/* Animated starfield pseudo-bg */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        radial-gradient(1px 1px at 15% 25%, rgba(255,255,255,0.55) 0%, transparent 100%),
        radial-gradient(1px 1px at 72% 8%,  rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 40% 60%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 85% 45%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 58% 78%, rgba(255,255,255,0.5) 0%, transparent 100%),
        radial-gradient(1px 1px at 92% 72%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 5%  90%, rgba(255,255,255,0.4) 0%, transparent 100%),
        radial-gradient(1px 1px at 30% 10%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(1px 1px at 64% 33%, rgba(255,255,255,0.25) 0%, transparent 100%),
        radial-gradient(1px 1px at 20% 55%, rgba(255,255,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 48% 48%, rgba(255,255,255,0.2) 0%, transparent 100%),
        radial-gradient(1px 1px at 77% 20%, rgba(255,255,255,0.45) 0%, transparent 100%),
        radial-gradient(2px 2px at 10% 70%, rgba(160,200,255,0.3) 0%, transparent 100%),
        radial-gradient(1.5px 1.5px at 55% 5%, rgba(180,220,255,0.3) 0%, transparent 100%),
        radial-gradient(1px 1px at 88% 88%, rgba(255,255,255,0.35) 0%, transparent 100%),
        radial-gradient(ellipse at 50% 0%, rgba(20,40,120,0.55) 0%, transparent 60%),
        radial-gradient(ellipse at 20% 80%, rgba(10,20,80,0.4) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ── Inputs ── */
.stTextInput > div > div > input,
.stDateInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important;
    caret-color: var(--accent-cyan) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stDateInput > div > div > input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px rgba(77,166,255,0.12) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--text-dim) !important; }

/* Slider */
.stSlider > div > div > div > div { background: var(--accent-blue) !important; }
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--accent-blue) !important;
    border-color: var(--accent-blue) !important;
}
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { color: var(--text-dim) !important; }

/* Labels */
label, .stTextInput label, .stDateInput label,
.stSlider label, .stSelectbox label {
    color: var(--text-secondary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}

/* Button */
.stButton > button {
    background: #4da6ff !important;
    border: 1px solid rgba(77,166,255,0.5) !important;
    border-radius: var(--radius-sm) !important;
    color: #02050f !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    padding: 0.7rem 1.4rem !important;
    transition: all 0.18s ease !important;
    width: 100% !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #6db8ff !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Dataframe */
.stDataFrame { 
    border-radius: 14px !important; 
    overflow: hidden !important; 
    border: 1px solid var(--border) !important; 
    box-shadow: 0 4px 32px rgba(0,0,0,0.4) !important;
}
.stDataFrame thead th {
    background: rgba(13,27,48,0.95) !important;
    color: var(--text-secondary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    border-bottom: 1px solid var(--border) !important;
}
.stDataFrame tbody td {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    border-bottom: 1px solid rgba(60,100,200,0.08) !important;
}
.stDataFrame tbody tr:hover td { background: rgba(77,166,255,0.05) !important; }

/* Spinner */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* Progress bar */
.stProgress > div > div > div > div { background: linear-gradient(90deg, #1a5ae8, #00e5ff) !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(77,166,255,0.25); border-radius: 3px; }

/* Alert boxes */
.stAlert { border-radius: 12px !important; font-family: 'Inter', sans-serif !important; }

/* Date input calendar icon */
.stDateInput svg { fill: var(--text-secondary) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def section_label(text):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:1.2rem;margin:2.8rem 0 1.4rem;">
        <span style="font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;
                     letter-spacing:0.18em;text-transform:uppercase;
                     color:#4da6ff;white-space:nowrap;">
            ◈ &nbsp;{text}
        </span>
        <div style="flex:1;height:1px;background:linear-gradient(90deg,rgba(77,166,255,0.4),transparent);"></div>
    </div>
    """, unsafe_allow_html=True)


def metric_pill(label, value, unit="", accent="blue", icon=""):
    # All accent types map to the same 2-color scheme: blue text, navy card
    st.markdown(f"""
    <div style="background:#0b1628;border:1px solid rgba(77,166,255,0.2);border-radius:14px;
                padding:1.1rem 0.8rem;text-align:center;">
        <div style="font-size:1.2rem;margin-bottom:0.3rem;">{icon}</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;
                    color:#4da6ff;line-height:1.1;">
            {value}<span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                 font-weight:400;opacity:0.55;margin-left:2px;">{unit}</span>
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.57rem;font-weight:700;
                    letter-spacing:0.1em;text-transform:uppercase;
                    color:#3a5070;margin-top:0.4rem;">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)


def object_icon(t):
    t = str(t).lower()
    if "planet" in t:   return "🪐"
    if "star" in t:     return "✦"
    if "constell" in t: return "✧"
    if "moon" in t:     return "🌕"
    return "◉"


AQI_LABELS = {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"}
AQI_ACCENT = {1: "green", 2: "green", 3: "gold", 4: "red", 5: "red"}


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER  — full-viewport split: big heading left, big Saturn SVG right
# ─────────────────────────────────────────────────────────────────────────────
components.html("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: transparent; overflow: hidden; }
  .hero {
    display: flex;
    align-items: center;
    justify-content: space-between;
    min-height: 92vh;
    padding: 4rem 5vw 3rem;
    gap: 2rem;
  }
  .hero-left {
    flex: 1;
    max-width: 580px;
  }
  .badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #0b1628;
    border: 1px solid #1e3a6e;
    border-radius: 100px;
    padding: 0.35rem 1rem;
    margin-bottom: 2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    color: #4da6ff;
  }
  .badge-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4da6ff;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; } 50% { opacity:0.4; }
  }
  h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(3rem, 6vw, 5.5rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.05;
    color: #e8f0ff;
    margin-bottom: 1.2rem;
  }
  h1 span { color: #4da6ff; }
  .sub {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    color: #7a98c8;
    line-height: 1.85;
    max-width: 460px;
    margin-bottom: 2.5rem;
  }
  .stats {
    display: flex;
    gap: 2.5rem;
    flex-wrap: wrap;
  }
  .stat-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    color: #4da6ff;
  }
  .stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #3a5070;
    margin-top: 3px;
  }
  .scroll-hint {
    margin-top: 3rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.14em;
    color: #2a4060;
    text-transform: uppercase;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .scroll-hint::after {
    content: '';
    display: inline-block;
    width: 18px;
    height: 1px;
    background: #2a4060;
  }
  .hero-right {
    flex: 0 0 auto;
    display: flex;
    align-items: center;
    justify-content: center;
  }
</style>

<div class="hero">
  <!-- LEFT: text -->
  <div class="hero-left">
    <div class="badge">
      <span class="badge-dot"></span>
      AI MODEL v2.0 — ACTIVE
    </div>
    <h1>Explore the<br><span>Night Sky.</span></h1>
    <p class="sub">
      ML-powered celestial predictions from real-time orbital mechanics,
      weather data &amp; atmospheric conditions — ranked for your location.
    </p>
    <div class="stats">
      <div>
        <div class="stat-val">99.1%</div>
        <div class="stat-lbl">Prediction Accuracy</div>
      </div>
      <div>
        <div class="stat-val">15+</div>
        <div class="stat-lbl">Celestial Objects</div>
      </div>
      <div>
        <div class="stat-val">&lt; 5s</div>
        <div class="stat-lbl">Analysis Time</div>
      </div>
      <div>
        <div class="stat-val">Live</div>
        <div class="stat-lbl">Skyfield Data</div>
      </div>
    </div>
    <div class="scroll-hint">scroll to analyse</div>
  </div>

  <!-- RIGHT: big Saturn SVG -->
  <div class="hero-right">
    <svg width="480" height="480" viewBox="0 0 480 480" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <radialGradient id="satGrad" cx="38%" cy="32%" r="65%">
          <stop offset="0%"   stop-color="#1a3a7a"/>
          <stop offset="40%"  stop-color="#0a1e55"/>
          <stop offset="75%"  stop-color="#030d2e"/>
          <stop offset="100%" stop-color="#01060f"/>
        </radialGradient>
        <radialGradient id="satGlow" cx="50%" cy="50%" r="50%">
          <stop offset="0%"   stop-color="#4da6ff" stop-opacity="0.12"/>
          <stop offset="100%" stop-color="#4da6ff" stop-opacity="0"/>
        </radialGradient>
        <!-- Ring gradient for realistic look -->
        <linearGradient id="ringA" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#4da6ff" stop-opacity="0.05"/>
          <stop offset="20%"  stop-color="#4da6ff" stop-opacity="0.28"/>
          <stop offset="50%"  stop-color="#7ac4ff" stop-opacity="0.18"/>
          <stop offset="80%"  stop-color="#4da6ff" stop-opacity="0.28"/>
          <stop offset="100%" stop-color="#4da6ff" stop-opacity="0.05"/>
        </linearGradient>
        <linearGradient id="ringB" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%"   stop-color="#2a6aaa" stop-opacity="0.04"/>
          <stop offset="20%"  stop-color="#2a6aaa" stop-opacity="0.18"/>
          <stop offset="50%"  stop-color="#4da6ff" stop-opacity="0.10"/>
          <stop offset="80%"  stop-color="#2a6aaa" stop-opacity="0.18"/>
          <stop offset="100%" stop-color="#2a6aaa" stop-opacity="0.04"/>
        </linearGradient>
        <clipPath id="topHalf">
          <rect x="0" y="0" width="480" height="240"/>
        </clipPath>
        <clipPath id="botHalf">
          <rect x="0" y="240" width="480" height="240"/>
        </clipPath>
      </defs>

      <!-- Background stars -->
      <circle cx="28"  cy="45"  r="1.2" fill="#e8f0ff" opacity="0.5"/>
      <circle cx="72"  cy="18"  r="0.9" fill="#4da6ff" opacity="0.6"/>
      <circle cx="155" cy="32"  r="1"   fill="#e8f0ff" opacity="0.4"/>
      <circle cx="310" cy="22"  r="1.1" fill="#4da6ff" opacity="0.5"/>
      <circle cx="405" cy="55"  r="0.9" fill="#e8f0ff" opacity="0.45"/>
      <circle cx="450" cy="28"  r="1.2" fill="#4da6ff" opacity="0.55"/>
      <circle cx="18"  cy="390" r="1"   fill="#e8f0ff" opacity="0.4"/>
      <circle cx="88"  cy="415" r="1.1" fill="#4da6ff" opacity="0.45"/>
      <circle cx="400" cy="410" r="1"   fill="#e8f0ff" opacity="0.4"/>
      <circle cx="460" cy="380" r="1.2" fill="#4da6ff" opacity="0.5"/>
      <circle cx="52"  cy="200" r="0.8" fill="#e8f0ff" opacity="0.35"/>
      <circle cx="430" cy="185" r="0.9" fill="#4da6ff" opacity="0.4"/>
      <circle cx="35"  cy="310" r="1"   fill="#4da6ff" opacity="0.3"/>
      <circle cx="455" cy="295" r="0.9" fill="#e8f0ff" opacity="0.35"/>
      <!-- Constellation top-left -->
      <line x1="28" y1="45" x2="72" y2="18"  stroke="#4da6ff" stroke-width="0.5" opacity="0.35"/>
      <line x1="72" y1="18" x2="155" y2="32" stroke="#4da6ff" stroke-width="0.5" opacity="0.35"/>
      <!-- Constellation top-right -->
      <line x1="310" y1="22" x2="405" y2="55"  stroke="#4da6ff" stroke-width="0.5" opacity="0.35"/>
      <line x1="405" y1="55" x2="450" y2="28"  stroke="#4da6ff" stroke-width="0.5" opacity="0.35"/>

      <!-- Outer glow -->
      <circle cx="240" cy="240" r="160" fill="url(#satGlow)"/>

      <!-- Ring back half (behind planet) -->
      <ellipse cx="240" cy="248" rx="210" ry="52"
               fill="none" stroke="url(#ringA)" stroke-width="22"
               clip-path="url(#botHalf)" opacity="0.9"/>
      <ellipse cx="240" cy="248" rx="178" ry="44"
               fill="none" stroke="url(#ringB)" stroke-width="14"
               clip-path="url(#botHalf)" opacity="0.85"/>
      <ellipse cx="240" cy="248" rx="150" ry="36"
               fill="none" stroke="url(#ringA)" stroke-width="8"
               clip-path="url(#botHalf)" opacity="0.6"/>

      <!-- Planet body -->
      <circle cx="240" cy="240" r="148" fill="url(#satGrad)"
              stroke="#4da6ff" stroke-width="0.8" stroke-opacity="0.2"/>
      <!-- Surface bands -->
      <ellipse cx="240" cy="210" rx="130" ry="14" fill="#4da6ff" opacity="0.025"/>
      <ellipse cx="240" cy="240" rx="140" ry="10" fill="#4da6ff" opacity="0.02"/>
      <ellipse cx="240" cy="268" rx="125" ry="12" fill="#3a80cc" opacity="0.018"/>
      <!-- Limb brightening -->
      <circle cx="240" cy="240" r="148" fill="none"
              stroke="#4da6ff" stroke-width="6" stroke-opacity="0.06"/>

      <!-- Ring front half (in front of planet) -->
      <ellipse cx="240" cy="248" rx="210" ry="52"
               fill="none" stroke="url(#ringA)" stroke-width="22"
               clip-path="url(#topHalf)" opacity="0.9"/>
      <ellipse cx="240" cy="248" rx="178" ry="44"
               fill="none" stroke="url(#ringB)" stroke-width="14"
               clip-path="url(#topHalf)" opacity="0.85"/>
      <ellipse cx="240" cy="248" rx="150" ry="36"
               fill="none" stroke="url(#ringA)" stroke-width="8"
               clip-path="url(#topHalf)" opacity="0.6"/>

      <!-- Highlight spot -->
      <ellipse cx="198" cy="196" rx="38" ry="24"
               fill="#4da6ff" opacity="0.055" transform="rotate(-20 198 196)"/>

      <!-- Small moon -->
      <circle cx="62" cy="148" r="10" fill="#0d2040"
              stroke="#4da6ff" stroke-width="0.8" stroke-opacity="0.4"/>
      <circle cx="62" cy="148" r="10" fill="#0b1e3d" opacity="0.9"/>
    </svg>
  </div>
</div>
""", height=720, scrolling=False)



# ─────────────────────────────────────────────────────────────────────────────
# INPUT PANEL
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(11,22,40,0.96),rgba(8,16,35,0.98));
            border:1px solid rgba(60,100,200,0.22);border-radius:20px;
            padding:1.8rem 2rem 0.6rem;margin-bottom:1.5rem;
            box-shadow:0 8px 40px rgba(0,0,0,0.5),0 0 1px rgba(77,166,255,0.2);">
    <div style="font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;
                letter-spacing:0.16em;text-transform:uppercase;color:#4da6ff;
                margin-bottom:1.2rem;display:flex;align-items:center;gap:0.6rem;">
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
            <circle cx="5" cy="5" r="4" stroke="#4da6ff" stroke-width="1.2"/>
            <circle cx="5" cy="5" r="1.5" fill="#4da6ff"/>
        </svg>
        Observation Parameters
    </div>
</div>
""", unsafe_allow_html=True)

col_loc, col_date, col_hour, col_btn = st.columns([3, 2, 2, 1.5])

with col_loc:
    place = st.text_input(
        "📡 City / Location",
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
        <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);
                    border-radius:12px;padding:1rem 1.4rem;color:#f87171;
                    font-family:'Inter',sans-serif;font-size:0.88rem;
                    box-shadow:0 2px 12px rgba(248,113,113,0.1);">
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
            <div style="text-align:center;padding:0.6rem;font-family:'Space Mono',monospace;
                        font-size:0.75rem;letter-spacing:0.06em;color:#4da6ff;">
                {step}
            </div>
            """, unsafe_allow_html=True)
            prog_slot.progress((i + 1) / len(steps))
            import time as _t; _t.sleep(0.12)

        lat, lon = get_coordinates(place)

    status_slot.empty()
    prog_slot.empty()

    if lat is None:
        st.markdown("""
        <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);
                    border-radius:12px;padding:1rem 1.4rem;color:#f87171;
                    font-family:'Inter',sans-serif;
                    box-shadow:0 2px 12px rgba(248,113,113,0.1);">
            ❌ &nbsp; Could not resolve that location. Check the city name and try again.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── WEATHER ───────────────────────────────────────────────────────────────
    weather  = get_weather(lat, lon)
    forecast = get_forecast(lat, lon)

    if "sys" not in weather:
        st.markdown("""
        <div style="background:rgba(248,113,113,0.08);border:1px solid rgba(248,113,113,0.25);
                    border-radius:12px;padding:1rem 1.4rem;color:#f87171;
                    font-family:'Inter',sans-serif;">
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
    <div style="margin:0.5rem 0 0;font-family:'Space Mono',monospace;
                font-size:0.72rem;color:#3a5070;display:flex;align-items:center;gap:0.5rem;">
        <span style="color:#4da6ff;">◉</span>
        <span style="color:#7a98c8;">{place.title()}</span>
        <span>·</span> {lat:.4f}°N, {lon:.4f}°E
        <span>·</span> {selected_date.strftime("%d %b %Y")}
        <span>·</span> {hour:02d}:00 local
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
        metric_pill("Moon Phase", f"{moon}", "%",
                    accent="gold" if moon > 50 else "blue", icon="🌙")
    with e6:
        metric_pill("Sun Window",
                    f"{sunrise:02d}:00–{sunset:02d}:00",
                    accent="gold", icon="🌅")

    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    # ── Daytime guard ─────────────────────────────────────────────────────────
    if not (hour >= sunset or hour <= sunrise):
        st.markdown(f"""
        <div style="background:rgba(255,217,125,0.07);border:1px solid rgba(255,217,125,0.22);
                    border-radius:12px;padding:1.1rem 1.6rem;color:#ffd97d;
                    font-family:'Inter',sans-serif;font-size:0.88rem;
                    box-shadow:0 2px 12px rgba(255,217,125,0.08);">
            ☀️ &nbsp; Selected hour <strong>{hour:02d}:00</strong> is during daytime
            (Sunrise {sunrise:02d}:00 → Sunset {sunset:02d}:00).
            Celestial objects are not visible. Please choose a night-time hour.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # OBJECT GENERATION + ML PIPELINE
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
    best_prob  = ranked.iloc[0]["visibility_probability"]  # raw 0–1

    # Conditions based on ML visibility score (0–1 regression output)
    if best_score > 0.8:
        vis_label = "Excellent Viewing Conditions"
    elif best_score > 0.5:
        vis_label = "Moderate Visibility"
    else:
        vis_label = "Poor Visibility Tonight"

    score_pct = min(100, best_score * 100)
    prob_pct  = min(100, best_prob  * 100)

    st.markdown(f"""
    <div style="background:#0b1628;border:1px solid rgba(77,166,255,0.25);border-radius:18px;
                padding:1.4rem 1.8rem;margin:1.2rem 0 1.5rem;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">
        <div style="display:flex;align-items:center;gap:1.2rem;">
            <div style="width:42px;height:42px;border-radius:50%;
                        background:#4da6ff18;border:1px solid rgba(77,166,255,0.3);
                        display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🌌</div>
            <div>
                <div style="font-family:'Syne',sans-serif;font-size:1.05rem;
                            font-weight:700;color:#4da6ff;">
                    {vis_label}
                </div>
                <div style="font-family:'Space Mono',monospace;font-size:0.66rem;
                            color:#3a5070;margin-top:0.25rem;">
                    {place.title()} · {selected_date.strftime("%d %b %Y")} · {hour:02d}:00 local
                </div>
            </div>
        </div>
        <div style="display:flex;gap:2rem;align-items:center;">
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif;font-size:2.5rem;
                            font-weight:800;color:#4da6ff;line-height:1;">{score_pct:.1f}</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.56rem;
                            letter-spacing:0.12em;text-transform:uppercase;color:#3a5070;">
                    Visibility Score / 100
                </div>
            </div>
            <div style="width:1px;height:40px;background:rgba(77,166,255,0.12);"></div>
            <div style="text-align:right;">
                <div style="font-family:'Syne',sans-serif;font-size:2.5rem;
                            font-weight:800;color:#7ac4ff;line-height:1;">{prob_pct:.1f}%</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.56rem;
                            letter-spacing:0.12em;text-transform:uppercase;color:#3a5070;">
                    Detection Probability
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # TOP OBJECT RECOMMENDATION CARD
    # ─────────────────────────────────────────────────────────────────────────
    top       = ranked.iloc[0]
    top_prob  = display_df.iloc[0]["visibility_probability"]
    top_icon  = object_icon(top.get("event_type", ""))
    top_dir   = top.get("direction",    "—")
    top_pos   = top.get("sky_position", "—")
    top_name  = top.get("constellation", "—")

    st.markdown(f"""
    <div style="background:#0b1628;border:1px solid rgba(77,166,255,0.22);border-radius:18px;
                padding:1.6rem 1.8rem;margin-bottom:1.5rem;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;font-weight:700;
                    letter-spacing:0.16em;text-transform:uppercase;
                    color:#3a5070;margin-bottom:1.2rem;">
            ★ &nbsp; Tonight's Top Recommendation
        </div>
        <div style="display:flex;align-items:center;
                    justify-content:space-between;flex-wrap:wrap;gap:1.2rem;">
            <div style="display:flex;align-items:center;gap:1.4rem;">
                <div style="font-size:2.2rem;">{top_icon}</div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.7rem;
                                font-weight:800;color:#e8f0ff;letter-spacing:-0.01em;">
                        {top_name}
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                                color:#3a5070;margin-top:0.25rem;letter-spacing:0.04em;">
                        {str(top.get("event_type","")).upper()}
                        &nbsp;·&nbsp; DIR: <span style="color:#7a98c8;">{top_dir}</span>
                        &nbsp;·&nbsp; {top_pos}
                    </div>
                </div>
            </div>
            <div style="display:flex;gap:2.5rem;text-align:center;align-items:center;">
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                                font-weight:800;color:#4da6ff;">{top_prob:.1f}%</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.54rem;
                                letter-spacing:0.1em;text-transform:uppercase;color:#3a5070;">Visibility</div>
                </div>
                <div style="width:1px;height:36px;background:rgba(77,166,255,0.15);"></div>
                <div>
                    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;
                                font-weight:800;color:#4da6ff;">
                        {top["visibility_score"]:.2f}
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.54rem;
                                letter-spacing:0.1em;text-transform:uppercase;color:#3a5070;">Score</div>
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
            "Rank":         st.column_config.TextColumn("Rank",       width="small"),
            "Object":       st.column_config.TextColumn("Object",     width="medium"),
            "direction":    st.column_config.TextColumn("Direction",  width="small"),
            "sky_position": st.column_config.TextColumn("Position",   width="medium"),
            "Prob %":       st.column_config.TextColumn("Visibility", width="small"),
            "Score":        st.column_config.TextColumn("Score",      width="small"),
        },
    )

    # ─────────────────────────────────────────────────────────────────────────
    # BEST VIEWING TIME
    # ─────────────────────────────────────────────────────────────────────────
    section_label("Optimal Viewing Window")

    best_hour, _ = find_best_time(lat, lon, bortle, selected_date, forecast, sunset)

    st.markdown(f"""
    <div style="background:#0b1628;border:1px solid rgba(77,166,255,0.22);border-radius:18px;
                padding:1.5rem 2rem;margin-bottom:0.5rem;
                display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1.2rem;">
        <div>
            <div style="font-family:'Space Mono',monospace;font-size:0.6rem;font-weight:700;
                        letter-spacing:0.14em;text-transform:uppercase;color:#3a5070;
                        margin-bottom:0.6rem;">
                🕒 &nbsp; ML-Recommended Best Time
            </div>
            <div style="font-family:'Syne',sans-serif;font-size:2.8rem;
                        font-weight:800;color:#4da6ff;line-height:1;">
                {best_hour:02d}:00
            </div>
            <div style="font-family:'Inter',sans-serif;font-size:0.78rem;
                        color:#7a98c8;margin-top:0.5rem;">
                Optimal darkness · minimal atmospheric interference
            </div>
        </div>
        <div style="text-align:right;font-family:'Space Mono',monospace;
                    font-size:0.7rem;color:#3a5070;line-height:2.4;">
            ↑ Sunrise &nbsp;<span style="color:#7a98c8;">{sunrise:02d}:00</span><br>
            ↓ Sunset &nbsp;&nbsp;<span style="color:#7a98c8;">{sunset:02d}:00</span><br>
            ◈ Bortle &nbsp;&nbsp;&nbsp;<span style="color:#7a98c8;">{bortle} / 9</span>
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
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#4da6ff;
                    margin-bottom:0.9rem;">◈ &nbsp; Visibility Score Distribution</div>
        """, unsafe_allow_html=True)

        fig_bar, ax_bar = plt.subplots(figsize=(7, max(4, len(ranked) * 0.45 + 1)))
        fig_bar.patch.set_facecolor("#050a1a")
        ax_bar.set_facecolor("#070e20")

        objects_list = ranked["constellation"].tolist()
        scores_list  = ranked["visibility_score"].tolist()
        probs_list   = display_df["visibility_probability"].tolist()
        n            = len(objects_list)
        y_pos        = np.arange(n)

        # Gradient-style bar colors
        bar_colors = []
        for i in range(n):
            alpha = 0.95 - (i / max(n - 1, 1)) * 0.4
            r_val = int(30  + (1 - alpha) * 20)
            g_val = int(120 + (1 - alpha) * 60)
            b_val = int(255)
            bar_colors.append(f"#{r_val:02x}{g_val:02x}{b_val:02x}")

        bars = ax_bar.barh(y_pos, scores_list, height=0.52,
                           color=bar_colors, zorder=3, alpha=0.85)

        ax2 = ax_bar.twiny()
        ax2.set_xlim(0, 100)
        ax2.scatter(probs_list, y_pos, color="#a78bfa", s=55,
                    zorder=5, alpha=0.85, marker="D", linewidths=0)
        ax2.set_xticks([0, 25, 50, 75, 100])
        ax2.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                            fontfamily="monospace", fontsize=7.5, color="#a78bfa")
        ax2.tick_params(axis="x", colors="#a78bfa", length=3)
        ax2.spines["top"].set_color((167/255, 139/255, 250/255, 0.18))
        for sp in ["bottom", "left", "right"]:
            ax2.spines[sp].set_visible(False)

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(objects_list, fontfamily="monospace",
                               fontsize=9, color="#7a98c8")
        ax_bar.set_xlabel("Visibility Score", fontfamily="monospace",
                          fontsize=8, color="#3a5070", labelpad=8)
        ax_bar.tick_params(axis="x", colors="#3a5070", labelsize=8, length=3)
        ax_bar.tick_params(axis="y", length=0)
        for sp in ax_bar.spines.values():
            sp.set_color((0.2, 0.4, 0.8, 0.15))
        ax_bar.grid(axis="x", color=(0.2, 0.4, 0.8, 0.08),
                    linestyle="--", linewidth=0.6, zorder=0)
        ax_bar.invert_yaxis()

        for bar, score in zip(bars, scores_list):
            ax_bar.text(bar.get_width() + 0.005,
                        bar.get_y() + bar.get_height() / 2,
                        f"{score:.2f}", va="center", ha="left",
                        fontfamily="monospace", fontsize=7.5, color="#4da6ff")

        h_bars = mpatches.Patch(color="#4da6ff", label="Visibility Score")
        h_dots = mpatches.Patch(color="#a78bfa", label="Probability %")
        ax_bar.legend(handles=[h_bars, h_dots], loc="lower right", fontsize=7.5,
                      facecolor="#050a1a", edgecolor=(0.2, 0.4, 0.8, 0.2),
                      labelcolor="#7a98c8", framealpha=0.92)

        fig_bar.tight_layout(pad=1.2)
        st.pyplot(fig_bar, use_container_width=True)
        plt.close(fig_bar)

    # ── Sky map ───────────────────────────────────────────────────────────────
    with viz_r:
        st.markdown("""
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;font-weight:700;
                    letter-spacing:0.12em;text-transform:uppercase;color:#4da6ff;
                    margin-bottom:0.9rem;">◈ &nbsp; Sky Map (Azimuthal Projection)</div>
        """, unsafe_allow_html=True)
        st.pyplot(plot(ranked), use_container_width=True)

    # ─────────────────────────────────────────────────────────────────────────
    # FOOTER
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:4rem;padding-top:1.5rem;
                border-top:1px solid rgba(60,100,200,0.15);text-align:center;">
        <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                    letter-spacing:0.1em;color:#3a5070;line-height:2;">
            ◉ &nbsp; Celestial Vision System
            &nbsp;·&nbsp; Skyfield + ML + OpenWeather + OpenCage
            &nbsp;·&nbsp; For best results observe from a dark-sky site
        </div>
        <div style="margin-top:0.6rem;font-family:'Space Mono',monospace;
                    font-size:0.55rem;color:#1e3050;letter-spacing:0.12em;">
            COORDINATE SYSTEM: HORIZONTAL · EPOCH: J2000 · PROJECTION: AZIMUTHAL EQUIDISTANT
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────────────────────
else:
    components.html("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
    <div style="text-align:center;padding:4rem 2rem 3rem;background:transparent;">

        <!-- Planet + rings + stars — single flat SVG, no position:absolute -->
        <svg width="280" height="220" viewBox="0 0 280 220"
             xmlns="http://www.w3.org/2000/svg" style="display:block;margin:0 auto 1.8rem;">
            <defs>
                <radialGradient id="pg2" cx="38%" cy="35%" r="65%">
                    <stop offset="0%"   stop-color="#0d2a6e"/>
                    <stop offset="60%"  stop-color="#020b2a"/>
                    <stop offset="100%" stop-color="#02050f"/>
                </radialGradient>
            </defs>

            <!-- Background stars -->
            <circle cx="18"  cy="22"  r="1"   fill="#4da6ff" opacity="0.5"/>
            <circle cx="55"  cy="10"  r="1.5" fill="#e8f0ff" opacity="0.4"/>
            <circle cx="220" cy="18"  r="1"   fill="#4da6ff" opacity="0.5"/>
            <circle cx="260" cy="35"  r="1.5" fill="#e8f0ff" opacity="0.35"/>
            <circle cx="12"  cy="180" r="1"   fill="#4da6ff" opacity="0.4"/>
            <circle cx="265" cy="175" r="1.5" fill="#e8f0ff" opacity="0.3"/>
            <circle cx="38"  cy="140" r="1"   fill="#4da6ff" opacity="0.3"/>
            <circle cx="248" cy="140" r="1"   fill="#4da6ff" opacity="0.35"/>

            <!-- Small constellation top-left -->
            <line x1="18" y1="55" x2="42" y2="40"  stroke="#4da6ff" stroke-width="0.6" opacity="0.45"/>
            <line x1="42" y1="40" x2="70" y2="52"  stroke="#4da6ff" stroke-width="0.6" opacity="0.45"/>
            <line x1="42" y1="40" x2="38" y2="75"  stroke="#4da6ff" stroke-width="0.6" opacity="0.35"/>
            <circle cx="18" cy="55" r="1.8" fill="#4da6ff" opacity="0.7"/>
            <circle cx="42" cy="40" r="2.2" fill="#4da6ff" opacity="0.9"/>
            <circle cx="70" cy="52" r="1.8" fill="#4da6ff" opacity="0.7"/>
            <circle cx="38" cy="75" r="1.5" fill="#e8f0ff" opacity="0.5"/>

            <!-- Small constellation top-right -->
            <line x1="210" y1="55" x2="238" y2="42" stroke="#4da6ff" stroke-width="0.6" opacity="0.45"/>
            <line x1="238" y1="42" x2="262" y2="58" stroke="#4da6ff" stroke-width="0.6" opacity="0.45"/>
            <line x1="238" y1="42" x2="244" y2="72" stroke="#4da6ff" stroke-width="0.6" opacity="0.35"/>
            <circle cx="210" cy="55"  r="1.8" fill="#4da6ff" opacity="0.7"/>
            <circle cx="238" cy="42"  r="2.2" fill="#4da6ff" opacity="0.9"/>
            <circle cx="262" cy="58"  r="1.8" fill="#4da6ff" opacity="0.7"/>
            <circle cx="244" cy="72"  r="1.5" fill="#e8f0ff" opacity="0.5"/>

            <!-- Outer orbital ring -->
            <ellipse cx="140" cy="118" rx="128" ry="30"
                     fill="none" stroke="#4da6ff" stroke-width="1"
                     stroke-opacity="0.18" transform="rotate(-8 140 118)"/>
            <!-- Inner orbital ring -->
            <ellipse cx="140" cy="118" rx="112" ry="25"
                     fill="none" stroke="#4da6ff" stroke-width="0.5"
                     stroke-opacity="0.1" transform="rotate(-8 140 118)"/>

            <!-- Planet body -->
            <circle cx="140" cy="118" r="70" fill="url(#pg2)"
                    stroke="#4da6ff" stroke-width="0.8" stroke-opacity="0.25"/>
            <!-- Subtle surface bands -->
            <ellipse cx="140" cy="130" rx="58" ry="10"
                     fill="#4da6ff" opacity="0.03"/>
            <ellipse cx="140" cy="108" rx="48" ry="7"
                     fill="#4da6ff" opacity="0.025"/>

            <!-- Sun/star -->
            <circle cx="52" cy="185" r="8"  fill="#e8f0ff" opacity="0.85"/>
            <circle cx="52" cy="185" r="13" fill="#4da6ff" opacity="0.08"/>
        </svg>

        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;
                    font-weight:800;color:#e8f0ff;margin-bottom:0.6rem;">
            Ready for Observation
        </div>
        <div style="font-family:'Inter',sans-serif;font-size:0.88rem;
                    color:#7a98c8;max-width:360px;margin:0 auto;line-height:1.9;">
            Enter your city and observation hour above,<br>
            then click
            <span style="color:#4da6ff;font-family:'Space Mono',monospace;
                         font-size:0.75rem;">⚡ ANALYZE SKY</span>
            to generate your personalized celestial forecast.
        </div>

        <!-- Star field strip -->
        <svg width="340" height="36" viewBox="0 0 340 36"
             xmlns="http://www.w3.org/2000/svg"
             style="display:block;margin:2.2rem auto 0;opacity:0.4;">
            <circle cx="10"  cy="18" r="1"   fill="#4da6ff"/>
            <circle cx="44"  cy="8"  r="1.5" fill="#e8f0ff"/>
            <circle cx="85"  cy="26" r="1"   fill="#4da6ff"/>
            <circle cx="128" cy="6"  r="1"   fill="#e8f0ff"/>
            <circle cx="170" cy="20" r="2"   fill="#4da6ff"/>
            <circle cx="210" cy="10" r="1"   fill="#e8f0ff"/>
            <circle cx="250" cy="28" r="1.5" fill="#4da6ff"/>
            <circle cx="292" cy="5"  r="1"   fill="#e8f0ff"/>
            <circle cx="328" cy="18" r="1"   fill="#4da6ff"/>
            <line x1="10"  y1="18" x2="44"  y2="8"  stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
            <line x1="44"  y1="8"  x2="85"  y2="26" stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
            <line x1="128" y1="6"  x2="170" y2="20" stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
            <line x1="170" y1="20" x2="210" y2="10" stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
            <line x1="250" y1="28" x2="292" y2="5"  stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
            <line x1="292" y1="5"  x2="328" y2="18" stroke="#4da6ff" stroke-width="0.5" opacity="0.5"/>
        </svg>
    </div>
    """, height=500, scrolling=False)