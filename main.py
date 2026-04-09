import pandas as pd
from src.models.predict import predict
from src.models.ranking import rank_objects

# MATCH TRAINING COLUMN NAMES EXACTLY
data = {
    "observer_latitude": [18.52]*5,
    "observer_longitude": [73.85]*5,
    "hour": [21]*5,
    "month": [4]*5,
    "day": [3]*5,
    "is_night": [1]*5,
    "elevation_deg": [60, 30, 45, 20, 70],
    "azimuth_deg": [120, 200, 150, 80, 300],
    "estimated_magnitude": [1.2, 4.5, 2.3, 5.8, 0.5],
    "moon_illumination_percent": [20]*5,
    "bortle_scale": [6]*5,
    "event_type": ["planet", "star", "planet", "asteroid", "planet"],
    "constellation": ["Orion", "Leo", "Scorpius", "Ursa Major", "Cygnus"]
}

df = pd.DataFrame(data)

# Predict
results = predict(df)

# Rank
top_objects = rank_objects(results)

print("\n🔭 Top Visible Objects:\n")
print(top_objects[["visibility_score", "visibility_probability"]])