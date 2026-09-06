import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN_PATH = "data/processed/train_data.csv"
TEST_PATH = "data/processed/test_data.csv"
MODEL_PATH = "models/classification/random_forest_model.pkl"
OUTPUT_DIR = "models/explainable_ai"
TOP_N = 10


def load_assets():
    """Load trained model and processed datasets."""
    model = joblib.load(MODEL_PATH)
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return model, X_train, y_train, X_test, y_test


def get_positive_class_shap(shap_values, class_index=1):
    """
    Normalize SHAP output for binary classification and return
    the positive class explanation in 2D or 1D form.
    """
    if isinstance(shap_values, list):
        shap_values = shap_values[class_index] if len(shap_values) > class_index else shap_values[0]

    shap_values = np.asarray(shap_values)

    # If shape is (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        class_index = class_index if shap_values.shape[-1] > class_index else 0
        shap_values = shap_values[:, :, class_index]

    return shap_values


def plot_global_summary(shap_values, X_sample, output_path):
    """Save SHAP summary plot."""
    shap_values = np.asarray(shap_values)
    X_sample = X_sample.copy()

    # Match dimensions if needed
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    n_features = min(shap_values.shape[1], X_sample.shape[1])
    shap_values = shap_values[:, :n_features]
    X_sample = X_sample.iloc[:, :n_features]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_local_explanation(feature_names, shap_values_row, output_path):
    """Save a bar chart of top SHAP contributions for one player."""
    shap_values_row = np.asarray(shap_values_row).reshape(-1)
    feature_names = list(feature_names)

    n = min(len(feature_names), len(shap_values_row))
    feature_names = feature_names[:n]
    shap_values_row = shap_values_row[:n]

    top_idx = np.argsort(np.abs(shap_values_row))[::-1][:TOP_N]

    ordered_features = [feature_names[int(i)] for i in top_idx][::-1]
    ordered_values = shap_values_row[top_idx][::-1]
    ordered_colors = ["#d62728" if v > 0 else "#2ca02c" for v in ordered_values]

    plt.figure(figsize=(10, 6))
    plt.barh(ordered_features, ordered_values, color=ordered_colors)
    plt.axvline(0, color="black", linewidth=1)
    plt.title("Top SHAP Feature Contributions")
    plt.xlabel("SHAP value")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def explain_prediction():
    """Explain one suspicious player prediction."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model, X_train, y_train, X_test, y_test = load_assets()
    feature_names = list(X_train.columns)

    # Use a small background sample for SHAP
    background_size = min(200, len(X_train))
    background = X_train.sample(n=background_size, random_state=42)

    # Select the player with the highest cheat probability
    cheat_probs = model.predict_proba(X_test)[:, 1]
    player_index = int(np.argmax(cheat_probs))
    player_row = X_test.iloc[[player_index]]
    actual_label = int(y_test.iloc[player_index])

    # Predict
    prediction = int(model.predict(player_row)[0])
    probabilities = model.predict_proba(player_row)[0]

    # SHAP explainer
    explainer = shap.TreeExplainer(model, data=background)
    shap_values_all = explainer.shap_values(X_test)
    shap_values_all = get_positive_class_shap(shap_values_all)

    shap_values_row = explainer.shap_values(player_row)
    shap_values_row = get_positive_class_shap(shap_values_row)
    shap_values_row = np.asarray(shap_values_row).reshape(-1)
    # Save global summary plot
    summary_plot_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
    plot_global_summary(shap_values_all, X_test, summary_plot_path)

    # Save local explanation plot
    local_plot_path = os.path.join(OUTPUT_DIR, "shap_player_explanation.png")
    plot_local_explanation(feature_names, shap_values_row, local_plot_path)

    # Prepare readable explanation
    top_indices = np.argsort(np.abs(shap_values_row))[::-1][:TOP_N]

    print("=" * 60)
    print("Explainable AI - SHAP Explanation")
    print("=" * 60)
    print(f"Selected Player Index : {player_index}")
    print(f"Actual Label          : {'Cheater' if actual_label == 1 else 'Fair Player'}")
    print(f"Predicted Label       : {'Cheater' if prediction == 1 else 'Fair Player'}")
    print(f"Fair Probability      : {probabilities[0] * 100:.2f}%")
    print(f"Cheat Probability     : {probabilities[1] * 100:.2f}%")
    print("\nTop Feature Contributions:")
    print("-" * 60)

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append("Explainable AI - SHAP Explanation")
    report_lines.append("=" * 60)
    report_lines.append(f"Selected Player Index : {player_index}")
    report_lines.append(f"Actual Label          : {'Cheater' if actual_label == 1 else 'Fair Player'}")
    report_lines.append(f"Predicted Label       : {'Cheater' if prediction == 1 else 'Fair Player'}")
    report_lines.append(f"Fair Probability      : {probabilities[0] * 100:.2f}%")
    report_lines.append(f"Cheat Probability     : {probabilities[1] * 100:.2f}%")
    report_lines.append("")
    report_lines.append("Top Feature Contributions:")

    for idx in top_indices:
        feature_name = feature_names[idx]
        shap_value = shap_values_row[idx]
        feature_value = player_row.iloc[0, idx]

        direction = "pushes toward CHEATER" if shap_value > 0 else "pushes toward FAIR"
        line = f"{feature_name:25s} | value = {feature_value:.4f} | SHAP = {shap_value:.4f} | {direction}"
        print(line)
        report_lines.append(line)

    # Save text report
    report_path = os.path.join(OUTPUT_DIR, "shap_explanation_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))

    print("\nSaved files:")
    print(f"- {summary_plot_path}")
    print(f"- {local_plot_path}")
    print(f"- {report_path}")
    print("\nExplainable AI step complete!")


if __name__ == "__main__":
    explain_prediction()