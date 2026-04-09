import sys
import os
import requests
import math
import random
import matplotlib.pyplot as plt
from datetime import datetime, date

from skyfield.api import load, Topos

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd

from src.models.predict import predict
from src.models.ranking import rank_objects


# -------------------------------
# API KEYS
# -------------------------------
OPENCAGE_API_KEY = "8910c80cb9c64b73b01e3db45bc27522"
WEATHER_API_KEY = "ee2e514941fda48295fd461a251082fb"


# -------------------------------
# SKYFIELD
# -------------------------------
ts = load.timescale()
planets = load('de421.bsp')


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


# -------------------------------
# UI
# -------------------------------
# -------------------------------
# UI (CLEAN + SAFE)
# -------------------------------

st.set_page_config(page_title="Celestial Visibility AI", layout="wide")

st.title("🌌 Celestial Visibility AI")
st.caption("AI-powered sky visibility prediction using ML + Astronomy")

# -------------------------------
# INPUT SECTION
# -------------------------------
with st.container():
    st.subheader("📍 Observation Settings")

    col1, col2, col3 = st.columns(3)

    with col1:
        place = st.text_input("Location", "Pune")

    with col2:
        selected_date = st.date_input("Date", date.today())

    with col3:
        hour = st.slider("Hour (24h)", 0, 23, 21)

# -------------------------------
# BUTTON
# -------------------------------
if st.button("🔭 Predict Visibility", use_container_width=True):

    # -------------------------------
    # LOCATION
    # -------------------------------
    lat, lon = get_coordinates(place)

    if lat is None:
        st.error("❌ Could not find location")
        st.stop()

    st.success(f"📍 {place} → Lat: {lat:.2f}, Lon: {lon:.2f}")

    # -------------------------------
    # WEATHER
    # -------------------------------
    weather = get_weather(lat, lon)
    forecast = get_forecast(lat, lon)

    if "sys" not in weather:
        st.error("❌ Weather API failed")
        st.stop()

    sunrise = datetime.fromtimestamp(weather["sys"]["sunrise"]).hour
    sunset = datetime.fromtimestamp(weather["sys"]["sunset"]).hour

    # -------------------------------
    # ENVIRONMENT PANEL
    # -------------------------------
    st.subheader("🌤️ Sky Conditions")

    col1, col2, col3, col4 = st.columns(4)

    clouds, humidity = extract_forecast(forecast, selected_date, hour)
    aqi = get_aqi(lat, lon)
    bortle = aqi_to_bortle(aqi)
    moon = get_moon(selected_date)

    col1.metric("☁️ Clouds", f"{clouds}%")
    col2.metric("💧 Humidity", f"{humidity}%")
    aqi_labels = {
    1: "Good",
    2: "Fair",
    3: "Moderate",
    4: "Poor",
    5: "Very Poor"
}

    col3.metric("🌫️ Air Quality", aqi_labels.get(aqi, "Unknown"))
    col4.metric("🌙 Moon", f"{moon}%")

    st.info(f"🌅 Sunrise: {sunrise}:00 | 🌇 Sunset: {sunset}:00")

    if not (hour >= sunset or hour <= sunrise):
        st.warning("☀️ Daytime — not visible")
        st.stop()

    # -------------------------------
    # OBJECT GENERATION
    # -------------------------------
    real = get_real_objects(lat, lon, selected_date, hour)
    stars = get_dataset_objects(selected_date)
    constellations = get_constellation_objects(selected_date)

    all_objs = real + stars + constellations

    df = pd.DataFrame(all_objs)

    df["observer_latitude"] = lat
    df["observer_longitude"] = lon
    df["hour"] = hour
    df["month"] = selected_date.month
    df["day"] = selected_date.day
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
    # RESULTS
    # -------------------------------
    st.subheader("🔭 Visible Objects")

    display_df = ranked.copy()
    display_df["visibility_probability"] = (
        display_df["visibility_probability"] * 100
    ).round(2)

    st.dataframe(display_df[[
        "event_type",
        "constellation",
        "direction",
        "sky_position",
        "visibility_score",
       # "visibility_probability"
    ]], use_container_width=True)

    # -------------------------------
    # INTERPRETATION
    # -------------------------------
    best = ranked.iloc[0]["visibility_score"]

    if best > 0.8:
        st.success("🌟 Excellent viewing conditions")
    elif best > 0.5:
        st.info("✨ Moderate visibility")
    else:
        st.warning("🌙 Poor visibility")

    # -------------------------------
    # BEST TIME
    # -------------------------------
    best_hour, _ = find_best_time(lat, lon, bortle, selected_date, forecast, sunset)

    st.subheader("🕒 Best Viewing Time")
    st.success(f"{best_hour}:00")

    # -------------------------------
    # VISUALS
    # -------------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Visibility Scores")
        st.bar_chart(ranked["visibility_score"])

    with col2:
        st.subheader("🌌 Sky Map")
        st.pyplot(plot(ranked))