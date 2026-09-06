import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

RAW_PATH = "data/raw/gameplay_data.csv"
PROCESSED_DIR = "data/processed"
SCALER_PATH = "models/saved_models/scaler.pkl"


def load_data(path):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def feature_engineering(df):
    df["kd_ratio"] = df["kills"] / (df["deaths"] + 1)
    df["headshot_ratio"] = df["headshots"] / (df["shots_hit"] + 1)
    df["hit_rate"] = df["shots_hit"] / (df["shots_fired"] + 1)
    return df


def preprocess_features(df):
    feature_cols = [
        "session_duration", "kills", "deaths", "assists",
        "shots_fired", "shots_hit", "accuracy", "headshots",
        "avg_reaction_time_ms", "avg_speed", "max_speed",
        "resources_collected", "damage_done",
        "suspicious_file_change", "kd_ratio", "headshot_ratio", "hit_rate"
    ]

    X = df[feature_cols]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    os.makedirs("models/saved_models", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)

    return X_scaled, y, feature_cols


def save_processed_data(X_train, X_test, y_train, y_test, feature_cols):
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df["label"] = y_train.reset_index(drop=True)

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df["label"] = y_test.reset_index(drop=True)

    train_df.to_csv(f"{PROCESSED_DIR}/train_data.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/test_data.csv", index=False)


def main():
    print("Loading raw data...")
    df = load_data(RAW_PATH)

    print("Cleaning data...")
    df = clean_data(df)

    print("Applying feature engineering...")
    df = feature_engineering(df)

    print("Preprocessing features...")
    X, y, feature_cols = preprocess_features(df)

    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, feature_cols)

    print("Preprocessing complete!")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")


if __name__ == "__main__":
    main()