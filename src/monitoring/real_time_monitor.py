import os
import time
import json
import random
from datetime import datetime

import pandas as pd
import joblib
import yaml


CONFIG_PATH = "config/config.yaml"
TEST_PATH = "data/processed/test_data.csv"
CLASSIFIER_PATH = "models/classification/random_forest_model.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_detection/isolation_forest_model.pkl"
LOG_PATH = "logs/alert_logs/realtime_monitoring_log.txt"


def load_config():
    """Load monitoring settings from config file."""
    default_config = {
        "monitoring": {
            "alert_threshold": 0.85,
            "check_interval": 1,
            "max_events": 10
        }
    }

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = yaml.safe_load(f) or {}
        default_config["monitoring"].update(cfg.get("monitoring", {}))

    return default_config


def load_assets():
    """Load classifier, anomaly model, and test dataset."""
    classifier = joblib.load(CLASSIFIER_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)

    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    return classifier, anomaly_model, X_test, y_test


def ensure_directories():
    """Ensure log directory exists."""
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)


def generate_live_event(X_test, y_test):
    """
    Simulate a live gameplay event by sampling from test data.
    Returns a player event and a separate file-integrity flag.
    """
    idx = random.randrange(len(X_test))
    event = X_test.iloc[[idx]].copy()
    actual_label = int(y_test.iloc[idx])

    player_id = random.randint(10000, 99999)
    match_id = random.randint(1000, 9999)

    # Simulate file-integrity status
    if actual_label == 1:
        file_integrity_flag = 1 if random.random() < 0.7 else 0
    else:
        file_integrity_flag = 1 if random.random() < 0.05 else 0

    return idx, player_id, match_id, event, actual_label, file_integrity_flag


def evaluate_event(classifier, anomaly_model, event, file_integrity_flag, alert_threshold):
    """Run classification + anomaly detection and compute a risk score."""
    cheat_prob = float(classifier.predict_proba(event)[0][1])
    pred_label = int(classifier.predict(event)[0])

    anomaly_pred = int(anomaly_model.predict(event)[0])
    anomaly_flag = 1 if anomaly_pred == -1 else 0

    risk_score = round(
        0.55 * cheat_prob +
        0.30 * anomaly_flag +
        0.15 * file_integrity_flag,
        4
    )

    if risk_score >= alert_threshold:
        action = "BLOCK MATCH ENTRY"
        severity = "HIGH"
    elif risk_score >= 0.65:
        action = "RESTRICT RANKED PLAY"
        severity = "MEDIUM"
    elif risk_score >= 0.50:
        action = "FLAG FOR ADMIN REVIEW"
        severity = "LOW"
    else:
        action = "ALLOW"
        severity = "SAFE"

    return {
        "predicted_label": pred_label,
        "cheat_prob": cheat_prob,
        "anomaly_flag": anomaly_flag,
        "file_integrity_flag": file_integrity_flag,
        "risk_score": risk_score,
        "action": action,
        "severity": severity
    }


def log_event(record):
    """Append monitoring event to log file."""
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def monitor_stream():
    """Simulate real-time monitoring of gameplay events."""
    config = load_config()
    alert_threshold = config["monitoring"]["alert_threshold"]
    check_interval = config["monitoring"]["check_interval"]
    max_events = int(config["monitoring"]["max_events"])

    classifier, anomaly_model, X_test, y_test = load_assets()
    ensure_directories()

    print("=" * 60)
    print("Real-Time Fair Play Monitoring System")
    print("=" * 60)
    print(f"Alert threshold : {alert_threshold}")
    print(f"Check interval  : {check_interval} sec")
    print(f"Max events      : {max_events}")
    print("-" * 60)

    for i in range(max_events):
        idx, player_id, match_id, event, actual_label, file_flag = generate_live_event(X_test, y_test)
        result = evaluate_event(classifier, anomaly_model, event, file_flag, alert_threshold)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        actual_text = "Cheater" if actual_label == 1 else "Fair Player"
        predicted_text = "Cheater" if result["predicted_label"] == 1 else "Fair Player"

        record = {
            "timestamp": timestamp,
            "event_no": i + 1,
            "player_id": player_id,
            "match_id": match_id,
            "sample_index": idx,
            "actual_label": actual_text,
            "predicted_label": predicted_text,
            "cheat_probability": round(result["cheat_prob"], 4),
            "anomaly_flag": result["anomaly_flag"],
            "file_integrity_flag": result["file_integrity_flag"],
            "risk_score": result["risk_score"],
            "severity": result["severity"],
            "action": result["action"]
        }

        log_event(record)

        print(f"[{timestamp}] Player {player_id} | Match {match_id}")
        print(f"  Actual: {actual_text} | Predicted: {predicted_text}")
        print(f"  Cheat Prob: {result['cheat_prob'] * 100:.2f}% | "
              f"Anomaly: {result['anomaly_flag']} | "
              f"File Flag: {file_flag}")
        print(f"  Risk Score: {result['risk_score']:.4f} | "
              f"Severity: {result['severity']} | "
              f"Action: {result['action']}")

        if result["action"] != "ALLOW":
            print("  ALERT: Suspicious activity detected!")

        print("-" * 60)
        time.sleep(check_interval)

    print("Real-time monitoring run complete.")


if __name__ == "__main__":
    monitor_stream()