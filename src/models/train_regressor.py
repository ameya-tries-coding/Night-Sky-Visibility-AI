import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    X = pd.read_csv("data/processed/X.csv")
    y = pd.read_csv("data/processed/y_reg.csv").values.ravel()
    return X, y


def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n📊 Regression Evaluation:")
    print("RMSE:", rmse)
    print("R2 Score:", r2)

    return y_pred


def save_model(model):
    joblib.dump(model, "models/regressor.pkl")
    print("✅ Regressor saved at models/regressor.pkl")


if __name__ == "__main__":
    print("🔄 Loading processed data...")

    X, y = load_data()

    print("📊 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🌲 Training Random Forest Regressor...")
    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    save_model(model)

    print("🚀 Regressor training complete!")