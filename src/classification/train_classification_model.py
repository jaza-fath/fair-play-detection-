import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve
)

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "data/processed/test_data.csv"
MODEL_PATH = "models/classification/random_forest_model.pkl"
PLOT_PATH = "models/classification/"


def load_data():
    """
    Load processed train and test data
    """
    print("Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    return train_df, test_df


def prepare_data(train_df, test_df):
    """
    Separate features and labels
    """
    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]

    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    print(f"Training samples  : {len(X_train)}")
    print(f"Testing samples   : {len(X_test)}")
    print(f"Features          : {X_train.shape[1]}")
    print(f"Fair players      : {(y_train == 0).sum()}")
    print(f"Cheaters          : {(y_train == 1).sum()}")

    return X_train, y_train, X_test, y_test


def train_model(X_train, y_train):
    """
    Train Random Forest Classifier
    """
    print("\nTraining Random Forest Classifier...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    print("Training complete!")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Evaluation Results")
    print("=" * 50)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"ROC AUC   : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Fair Player", "Cheater"]
    ))

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    """
    Plot and save confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fair Player", "Cheater"],
        yticklabels=["Fair Player", "Cheater"]
    )
    plt.title("Confusion Matrix - Fair Play Detection")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    os.makedirs(PLOT_PATH, exist_ok=True)
    plt.savefig(f"{PLOT_PATH}confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to: {PLOT_PATH}confusion_matrix.png")


def plot_roc_curve(y_test, y_prob):
    """
    Plot and save ROC curve
    """
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr,
        color="blue",
        label=f"ROC Curve (AUC = {roc_auc:.4f})"
    )
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Fair Play Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(f"{PLOT_PATH}roc_curve.png")
    plt.close()
    print(f"ROC curve saved to: {PLOT_PATH}roc_curve.png")


def plot_feature_importance(model, feature_cols):
    """
    Plot and save feature importance
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.bar(
        range(len(feature_cols)),
        importances[indices],
        color="steelblue"
    )
    plt.xticks(
        range(len(feature_cols)),
        [feature_cols[i] for i in indices],
        rotation=45,
        ha="right"
    )
    plt.title("Feature Importance - Fair Play Detection")
    plt.xlabel("Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()

    plt.savefig(f"{PLOT_PATH}feature_importance.png")
    plt.close()
    print(f"Feature importance saved to: {PLOT_PATH}feature_importance.png")


def save_model(model):
    """
    Save trained model
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")


def predict_player(model, player_data, feature_cols):
    """
    Predict if a single player is cheating
    """
    player_df = pd.DataFrame([player_data], columns=feature_cols)
    prediction = model.predict(player_df)[0]
    probability = model.predict_proba(player_df)[0]

    result = {
        "prediction": "Cheater" if prediction == 1 else "Fair Player",
        "confidence": f"{max(probability) * 100:.2f}%",
        "fair_probability": f"{probability[0] * 100:.2f}%",
        "cheat_probability": f"{probability[1] * 100:.2f}%"
    }

    return result


def main():
    print("=" * 50)
    print("Classification Model - Fair Play Detection")
    print("=" * 50)

    # Load data
    train_df, test_df = load_data()

    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data(train_df, test_df)

    feature_cols = list(X_train.columns)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    y_pred, y_prob = evaluate_model(model, X_test, y_test)

    # Plot results
    print("\nGenerating plots...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    plot_feature_importance(model, feature_cols)

    # Save model
    save_model(model)

    # Test single player prediction
    print("\nTesting single player prediction...")
    print("-" * 50)

    # Example fair player
    fair_player = [1800, 10, 8, 5, 150, 60, 0.4, 8,
                   240.0, 5.5, 8.0, 200, 2000, 0,
                   1.22, 0.13, 0.39]

    # Example cheater
    cheater = [1800, 45, 2, 3, 180, 160, 0.89, 140,
               75.0, 9.8, 15.0, 1200, 9000, 1,
               22.5, 0.88, 0.89]

    fair_result = predict_player(model, fair_player, feature_cols)
    cheat_result = predict_player(model, cheater, feature_cols)

    print("\nFair Player Prediction:")
    for key, value in fair_result.items():
        print(f"  {key}: {value}")

    print("\nSuspicious Player Prediction:")
    for key, value in cheat_result.items():
        print(f"  {key}: {value}")

    print("\nClassification model training complete!")


if __name__ == "__main__":
    main()