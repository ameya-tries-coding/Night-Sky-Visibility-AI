# 🌌 Celestial Night Sky Visibility AI

> **AI-powered night sky prediction** — Know exactly when, where, and what celestial objects you can see tonight.

Celestial Visibility AI combines **machine learning**, **real-time astronomy**, and **live weather data** to predict the visibility of planets, stars, and constellations from any location on Earth. Built with Streamlit, it provides an interactive dashboard with sky maps, visibility rankings, and optimal observation time recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📍 **Location Geocoding** | Resolve any city or place name to GPS coordinates via OpenCage |
| 🌤️ **Live Weather Integration** | Fetches real-time cloud cover, humidity, and air quality (AQI) from OpenWeatherMap |
| 🪐 **Real-Time Planetary Positions** | Uses NASA's DE421 ephemeris via Skyfield to compute planet altitudes and azimuths |
| ⭐ **Star & Constellation Data** | Simulates seasonal positions for major stars (Sirius, Vega, Rigel, Polaris, Betelgeuse) and constellations |
| 🤖 **ML Visibility Prediction** | A trained regressor predicts a `visibility_score` for each celestial object |
| 🏆 **Object Ranking** | Objects are ranked by AI-predicted visibility, adjusted for live cloud/humidity conditions |
| 🕒 **Best Time Finder** | Scans post-sunset hours to recommend the optimal observation window |
| 🌌 **Polar Sky Map** | Renders an interactive polar chart showing each object's position in the sky |
| 📊 **Visibility Score Chart** | Bar chart comparing all visible objects ranked by score |

---

## 🏗️ Project Structure

```
celestial-night-sky-visibility-ai/
│
├── app.py                    # Main Streamlit application (root entry point)
├── main.py                   # Alternate / utility entry point
├── celestial_app.py          # Extended / experimental version of the app
├── requirements.txt          # Python dependencies
├── API.txt                   # API key reference notes
│
├── frontend/                 # Frontend Streamlit variants
│   ├── app.py                # Frontend copy of the main app
│   ├── app_copy.py           # Experimental extended UI
│   ├── final.py              # Final polished frontend version
│   └── .streamlit/           # Streamlit config (theme, port, etc.)
│
├── src/                      # Core source modules
│   ├── models/               # ML model training and inference
│   │   ├── predict.py        # Loads models and generates predictions
│   │   ├── ranking.py        # Ranks objects by visibility score
│   │   ├── train_regressor.py  # Trains the gradient-boosted regressor
│   │   └── train_classifier.py # Trains the visibility classifier
│   ├── astronomy/            # Astronomy utilities (Skyfield helpers)
│   ├── api/                  # API wrapper functions (weather, geocoding)
│   ├── data/                 # Data processing utilities
│   └── utils/                # General helper utilities
│
├── models/                   # Serialized trained models
│   ├── regressor.pkl         # Gradient Boosted Regressor (~88 MB)
│   ├── classifier.pkl        # Random Forest Classifier (~10 MB)
│   └── encoder.pkl           # Label encoder for categorical features
│
├── data/                     # Datasets used for training
├── notebooks/                # Jupyter notebooks for EDA & prototyping
└── outputs/                  # Generated charts and output files
```

---

## 🤖 Machine Learning Pipeline

The visibility prediction is powered by two trained scikit-learn models:

### Regressor — `models/regressor.pkl`
- **Type**: Gradient Boosted Regressor
- **Target**: `visibility_score` (continuous, 0–1)
- **Features**: `observer_latitude`, `observer_longitude`, `hour`, `month`, `day`, `is_night`, `elevation_deg`, `azimuth_deg`, `estimated_magnitude`, `moon_illumination_percent`, `bortle_scale`, `event_type`, `constellation`

### Classifier — `models/classifier.pkl`
- **Type**: Random Forest Classifier
- **Target**: `visibility_probability` (binary / probability of being visible)
- **Features**: Same as above

### Real-Time Score Adjustment
Raw ML scores are post-processed to factor in live conditions:
```
adjusted_score = ml_score × ((100 - clouds) / 100) × ((100 - humidity) / 100)
```

### Bortle Scale Estimation
Air Quality Index (AQI) from OpenWeatherMap is mapped to the **Bortle Dark-Sky Scale**:

| AQI | Bortle | Sky Quality |
|-----|--------|-------------|
| 1   | 3      | Rural sky   |
| 2   | 4      | Rural/suburban transition |
| 3   | 5      | Suburban sky |
| 4   | 7      | Suburban/urban transition |
| 5   | 9      | Inner-city sky |

---

## 🚀 Getting Started

### Prerequisites
- Python **3.9+**
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/celestial-night-sky-visibility-ai.git
cd celestial-night-sky-visibility-ai
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
The app uses two external APIs. Open `app.py` and replace the placeholder keys:

```python
OPENCAGE_API_KEY = "your_opencage_api_key"    # For geocoding (place → lat/lon)
WEATHER_API_KEY  = "your_openweathermap_key"  # For weather, AQI, and forecast
```

> **Get your free keys:**
> - OpenCage Geocoder: https://opencagedata.com/api
> - OpenWeatherMap: https://openweathermap.org/api

### 5. Download the Ephemeris File
Skyfield requires NASA's `de421.bsp` planetary ephemeris. It downloads automatically on first run, or you can pre-download it:
```bash
python -c "from skyfield.api import load; load('de421.bsp')"
```

### 6. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🖥️ Usage

1. **Enter a Location** — Type any city name (e.g., "Pune", "London", "New York")
2. **Select a Date** — Pick any upcoming date
3. **Set the Hour** — Use the slider to choose a specific hour (24h format)
4. **Click "🔭 Predict Visibility"** — The app will:
   - Resolve your location to GPS coordinates
   - Fetch live weather and air quality
   - Compute real-time planetary positions using Skyfield
   - Run ML inference to score all visible objects
   - Display a ranked table, bar chart, and polar sky map
   - Recommend the **best observation hour** of the night

---

## 🧪 Training the Models

> [!IMPORTANT]
> **`models/regressor.pkl` has been removed** from this repository (file size ~88 MB).
> You **must retrain the regressor before running the app**, otherwise predictions will fail.
> Run the command below first:
> ```bash
> python src/models/train_regressor.py
> ```

To retrain all models on your own dataset:

```bash
# Train the regressor (required — model file not included)
python src/models/train_regressor.py

# Train the classifier
python src/models/train_classifier.py
```

Trained models are saved to `models/` as `.pkl` files.

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Interactive web UI |
| `skyfield` | Astronomical position calculations |
| `scikit-learn` | ML model training and inference |
| `pandas` | Data manipulation |
| `matplotlib` | Sky map (polar chart) and bar charts |
| `requests` | HTTP calls to weather and geocoding APIs |

---

## 🌍 External APIs

| API | Usage | Free Tier |
|---|---|---|
| [OpenCage Geocoding](https://opencagedata.com/) | City name → latitude/longitude | 2,500 requests/day |
| [OpenWeatherMap](https://openweathermap.org/) | Current weather, 5-day forecast, AQI | 1,000 calls/day |

---

## 🗺️ Sky Map Explained

The polar chart represents the **observer's view of the sky dome**:
- **Center** = Zenith (directly overhead)
- **Edge** = Horizon
- **North** = 0° (top of chart)
- **Clockwise** = East → South → West
- Each dot is a celestial object labeled with its name

---

## 📋 Roadmap

- [ ] Deep sky objects (nebulae, galaxies) via Messier catalog
- [ ] Multi-night forecast visualization
- [ ] ISS and satellite pass tracking
- [ ] User accounts with saved observation history
- [ ] Export observation plan as PDF
- [ ] Mobile-responsive layout

---

## 🤝 Contributing

Contributions are welcome! Please open an issue first to discuss your proposed changes.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---
<div align="center">
  <sub>Built with ❤️, Python, and a love for the night sky 🌠</sub>
</div>
