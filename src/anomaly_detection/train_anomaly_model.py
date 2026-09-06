import os
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "data/processed/test_data.csv"
MODEL_PATH = "models/anomaly_detection/isolation_forest_model.pkl"


def load_data():
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_data(train_df, test_df):
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    # Train only on fair player behavior
    normal_data = X_train[y_train == 0]

    model = IsolationForest(
        n_estimators=100,
        contamination=0.2,
        random_state=42
    )

    model.fit(normal_data)
    return model


def evaluate_model(model, X_test, y_test):
    raw_preds = model.predict(X_test)

    # IsolationForest output:
    # 1 = normal
    # -1 = anomaly
    y_pred = [0 if pred == 1 else 1 for pred in raw_preds]

    print("\nModel Evaluation")
    print("=" * 50)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def save_model(model):
    os.makedirs("models/anomaly_detection", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def main():
    print("Loading processed data...")
    train_df, test_df = load_data()

    print("Preparing features and labels...")
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)

    print("Training Isolation Forest model...")
    model = train_model(X_train, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("Saving model...")
    save_model(model)

    print("\nAnomaly detection training complete!")


if __name__ == "__main__":
    main()