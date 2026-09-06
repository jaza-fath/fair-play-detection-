import os
import sqlite3
import json
from datetime import datetime

DB_PATH = "data/database/fairplay.db"
REALTIME_LOG_PATH = "logs/alert_logs/realtime_monitoring_log.txt"


class DatabaseManager:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.initialize_database()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def initialize_database(self):
        """Create tables if they do not exist."""
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS player_activity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    player_id INTEGER,
                    match_id INTEGER,
                    sample_index INTEGER,
                    actual_label TEXT,
                    predicted_label TEXT,
                    cheat_probability REAL,
                    anomaly_flag INTEGER,
                    file_integrity_flag INTEGER,
                    risk_score REAL,
                    severity TEXT,
                    action TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    player_id INTEGER,
                    match_id INTEGER,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    risk_score REAL,
                    action TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS file_integrity_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    file_path TEXT,
                    stored_hash TEXT,
                    current_hash TEXT,
                    status TEXT,
                    note TEXT
                )
            """)

            conn.commit()
        print(f"Database initialized at: {self.db_path}")

    def save_player_activity(self, record):
        """Insert one player activity event."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO player_activity (
                    timestamp, player_id, match_id, sample_index,
                    actual_label, predicted_label, cheat_probability,
                    anomaly_flag, file_integrity_flag, risk_score,
                    severity, action
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                record.get("player_id"),
                record.get("match_id"),
                record.get("sample_index"),
                record.get("actual_label"),
                record.get("predicted_label"),
                record.get("cheat_probability"),
                record.get("anomaly_flag"),
                record.get("file_integrity_flag"),
                record.get("risk_score"),
                record.get("severity"),
                record.get("action")
            ))
            conn.commit()

    def save_alert(self, record):
        """Insert alert record when a suspicious player is detected."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (
                    timestamp, player_id, match_id, alert_type,
                    message, severity, risk_score, action
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                record.get("player_id"),
                record.get("match_id"),
                record.get("alert_type", "CHEAT_DETECTION"),
                record.get("message"),
                record.get("severity"),
                record.get("risk_score"),
                record.get("action")
            ))
            conn.commit()

    def save_file_integrity_record(self, record):
        """Insert file integrity result."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO file_integrity_records (
                    timestamp, file_path, stored_hash,
                    current_hash, status, note
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                record.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                record.get("file_path"),
                record.get("stored_hash"),
                record.get("current_hash"),
                record.get("status"),
                record.get("note")
            ))
            conn.commit()

    def import_realtime_log(self, log_path=REALTIME_LOG_PATH):
        """
        Import Step 7 real-time monitoring JSON log lines into the database.
        """
        if not os.path.exists(log_path):
            print(f"No realtime log found at: {log_path}")
            return 0

        imported = 0

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)

                    self.save_player_activity(record)
                    imported += 1

                    if record.get("action") != "ALLOW":
                        self.save_alert({
                            "timestamp": record.get("timestamp"),
                            "player_id": record.get("player_id"),
                            "match_id": record.get("match_id"),
                            "alert_type": "REAL_TIME_CHEAT_ALERT",
                            "message": f"Suspicious player detected. Action: {record.get('action')}",
                            "severity": record.get("severity"),
                            "risk_score": record.get("risk_score"),
                            "action": record.get("action")
                        })

                except json.JSONDecodeError:
                    print(f"Skipping invalid log line: {line[:80]}...")

        print(f"Imported {imported} realtime monitoring records.")
        return imported

    def get_recent_alerts(self, limit=10):
        """Fetch recent alerts."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM alerts
                ORDER BY id DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_player_activity(self, player_id, limit=10):
        """Fetch activity for a specific player."""
        with self.connect() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM player_activity
                WHERE player_id = ?
                ORDER BY id DESC
                LIMIT ?
            """, (player_id, limit))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_summary_counts(self):
        """Return total counts for all tables."""
        with self.connect() as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM player_activity")
            player_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM alerts")
            alert_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM file_integrity_records")
            integrity_count = cursor.fetchone()[0]

        return {
            "player_activity": player_count,
            "alerts": alert_count,
            "file_integrity_records": integrity_count
        }


def main():
    print("=" * 60)
    print("Fair Play Detection Database Integration")
    print("=" * 60)

    db = DatabaseManager()

    # 1) Import Step 7 realtime monitoring logs if available
    print("\n[1] Importing realtime monitoring logs...")
    db.import_realtime_log()

    # 2) Insert a sample file integrity record
    print("\n[2] Inserting sample file integrity record...")
    db.save_file_integrity_record({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "file_path": "game_files/anti_cheat.dll",
        "stored_hash": "abc123storedhash",
        "current_hash": "xyz789currenthash",
        "status": "MODIFIED",
        "note": "Simulated tampering detected for testing."
    })

    # 3) Show summary
    print("\n[3] Database summary:")
    summary = db.get_summary_counts()
    print(f"Player Activity Records : {summary['player_activity']}")
    print(f"Alert Records           : {summary['alerts']}")
    print(f"Integrity Records        : {summary['file_integrity_records']}")

    # 4) Show recent alerts
    print("\n[4] Recent alerts:")
    recent_alerts = db.get_recent_alerts(limit=5)

    if not recent_alerts:
        print("No alerts found.")
    else:
        for alert in recent_alerts:
            print(
                f"- [{alert['timestamp']}] Player {alert['player_id']} | "
                f"Severity: {alert['severity']} | Action: {alert['action']}"
            )

    print("\nStep 8 complete!")


if __name__ == "__main__":
    main()