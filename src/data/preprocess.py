import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):
    # -------------------------------
    # DROP USELESS COLUMN
    # -------------------------------
    if 'object_name' in df.columns:
        df = df.drop(columns=['object_name'])

    # -------------------------------
    # CONVERT DATE → FEATURES
    # -------------------------------
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df = df.drop(columns=['date'])

    # -------------------------------
    # ADD NIGHT FEATURE
    # -------------------------------
    df['is_night'] = df['hour'].apply(lambda x: 1 if x >= 18 or x <= 6 else 0)

    # -------------------------------
    # ADD REALISTIC NOISE (IMPORTANT)
    # -------------------------------
    np.random.seed(42)

    # Add noise to key features
    if 'elevation_deg' in df.columns:
        df['elevation_deg'] += np.random.normal(0, 3, len(df))

    if 'apparent_magnitude' in df.columns:
        df['apparent_magnitude'] += np.random.normal(0, 0.5, len(df))

    # Add noise to visibility score
    df['visibility_score'] += np.random.normal(0, 2, len(df))
    df['visibility_score'] = df['visibility_score'].clip(0, 100)

    # -------------------------------
    # REDEFINE TARGET (IMPORTANT)
    # -------------------------------
    df['visible_label'] = (df['visibility_score'] > 20).astype(int)

    # -------------------------------
    # SPLIT FEATURES & TARGETS
    # -------------------------------
    X = df.drop(columns=['visible_label', 'visibility_score'])
    y_class = df['visible_label']
    y_reg = df['visibility_score']

    # -------------------------------
    # ONE HOT ENCODING
    # -------------------------------
    categorical_cols = ['event_type', 'constellation']

    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

    encoded = encoder.fit_transform(X[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Drop old categorical columns
    X = X.drop(columns=categorical_cols)

    # Combine
    X = pd.concat([X.reset_index(drop=True), encoded_df], axis=1)

    return X, y_class, y_reg, encoder


if __name__ == "__main__":
    print("🔄 Loading data...")

    df = load_data("data/raw/astronomical_visibility_dataset_10k.csv")

    print("⚙️ Preprocessing...")

    X, y_class, y_reg, encoder = preprocess_data(df)

    # -------------------------------
    # SAVE FILES
    # -------------------------------
    X.to_csv("data/processed/X.csv", index=False)
    y_class.to_csv("data/processed/y_class.csv", index=False)
    y_reg.to_csv("data/processed/y_reg.csv", index=False)

    joblib.dump(encoder, "models/encoder.pkl")

    print("✅ Preprocessing complete!")
    print(f"📊 Feature shape: {X.shape}")