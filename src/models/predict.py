import pandas as pd
import joblib

classifier = joblib.load("models/classifier.pkl")
regressor = joblib.load("models/regressor.pkl")
encoder = joblib.load("models/encoder.pkl")

feature_columns = pd.read_csv("data/processed/X.csv").columns


def align_features(df):
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


def preprocess_input(input_df):
    categorical_cols = ['event_type', 'constellation']

    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    input_df = input_df.drop(columns=categorical_cols)
    input_df = pd.concat([input_df.reset_index(drop=True), encoded_df], axis=1)

    input_df = align_features(input_df)

    return input_df


def predict(input_df):
    # 🌞 DAYTIME RULE (OVERRIDE ML)
    if input_df["is_night"].iloc[0] == 0:
        results = input_df.copy()
        results["visible_label"] = 0
        results["visibility_probability"] = 0.0
        results["visibility_score"] = 0.0
        return results

    X = preprocess_input(input_df)

    visibility_class = classifier.predict(X)
    visibility_prob = classifier.predict_proba(X)[:, 1]
    visibility_score = regressor.predict(X)

    results = input_df.copy()
    results["visible_label"] = visibility_class
    results["visibility_probability"] = visibility_prob
    results["visibility_score"] = visibility_score

    return results