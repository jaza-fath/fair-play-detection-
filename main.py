import os
import sys
import json
import sqlite3
import subprocess
from datetime import datetime
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import joblib

from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go

app = Flask(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "database", "fairplay.db")

# -----------------------------
# Database Helpers
# -----------------------------
def get_db_stats():
    if not os.path.exists(DB_PATH):
        return {"players": 0, "alerts": 0, "high_alerts": 0, "integrity": 0}
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM player_activity")
        players = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts")
        alerts = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM alerts WHERE severity='HIGH'")
        high_alerts = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM file_integrity_records")
        integrity = c.fetchone()[0]
        conn.close()
        return {"players": players, "alerts": alerts, "high_alerts": high_alerts, "integrity": integrity}
    except Exception:
        return {"players": 0, "alerts": 0, "high_alerts": 0, "integrity": 0}

def get_recent_alerts():
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 8")
        rows = [dict(r) for r in c.fetchall()]
        conn.close()
        return rows
    except Exception:
        return []

# -----------------------------
# Embedded Dash App for Analytics
# -----------------------------
dash_app = Dash(
    __name__,
    server=app,
    url_base_pathname='/dashboard/',
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

def load_dash_data():
    if not os.path.exists(DB_PATH):
        return pd.DataFrame(), pd.DataFrame()
    try:
        conn = sqlite3.connect(DB_PATH)
        p_df = pd.read_sql_query("SELECT * FROM player_activity ORDER BY id DESC LIMIT 100", conn)
        a_df = pd.read_sql_query("SELECT * FROM alerts ORDER BY id DESC LIMIT 100", conn)
        conn.close()
        return p_df, a_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

dash_app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H2("📊 Fair Play Analytics Dashboard", className="mt-4 text-primary"),
            html.P("Real-time telemetry analytics and risk visualization.", className="text-muted"),
            html.A("⬅ Back to Portal", href="/", className="btn btn-outline-secondary btn-sm mb-3")
        ], md=12)
    ]),
    dcc.Interval(id="dash-refresh", interval=5000, n_intervals=0),
    dbc.Row([
        dbc.Col(dcc.Graph(id="dash-severity-chart"), md=6),
        dbc.Col(dcc.Graph(id="dash-action-chart"), md=6),
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.H4("Recent Alert Logs"),
            html.Div(id="dash-alerts-table-container")
        ], md=12)
    ])
], fluid=True)

@dash_app.callback(
    [Output("dash-severity-chart", "figure"),
     Output("dash-action-chart", "figure"),
     Output("dash-alerts-table-container", "children")],
    [Input("dash-refresh", "n_intervals")]
)
def update_dash_graphs(_):
    p_df, a_df = load_dash_data()

    # Severity chart
    if a_df.empty or "severity" not in a_df.columns:
        fig1 = go.Figure().update_layout(title="Alert Severity Distribution (No Data)")
    else:
        counts = a_df["severity"].value_counts()
        fig1 = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.4)])
        fig1.update_layout(title="Alert Severity Distribution", template="plotly_white")

    # Action chart
    if p_df.empty or "action" not in p_df.columns:
        fig2 = go.Figure().update_layout(title="Recommended Actions (No Data)")
    else:
        counts2 = p_df["action"].value_counts()
        fig2 = go.Figure(data=[go.Bar(x=counts2.index, y=counts2.values, marker_color="#1f77b4")])
        fig2.update_layout(title="Recommended Actions Distribution", template="plotly_white")

    # Table
    if a_df.empty:
        table = html.P("No alerts recorded yet.", className="text-muted")
    else:
        table = dbc.Table.from_dataframe(
            a_df[["timestamp", "player_id", "severity", "action", "message"]].head(10),
            striped=True, bordered=True, hover=True
        )

    return fig1, fig2, table

# -----------------------------
# Main Flask HTML Template
# -----------------------------
MAIN_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AI Fair Play Detection System | Esports Portal</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        body { background-color: #0d1117; color: #c9d1d9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        .navbar { background-color: #161b22; border-bottom: 1px solid #30363d; }
        .card { background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; margin-bottom: 20px; }
        .card-header { background-color: #21262d; border-bottom: 1px solid #30363d; font-weight: 600; color: #58a6ff; }
        .stat-card { border-left: 4px solid #58a6ff; }
        .stat-card.danger { border-left-color: #f85149; }
        .stat-card.warning { border-left-color: #d29922; }
        .stat-card.success { border-left-color: #3fb950; }
        .stat-val { font-size: 2rem; font-weight: bold; color: #f0f6fc; }
        .badge-HIGH { background-color: #f85149; color: white; }
        .badge-MEDIUM { background-color: #d29922; color: black; }
        .badge-SAFE, .badge-LOW { background-color: #3fb950; color: white; }
        .btn-custom { background-color: #238636; color: white; border: none; }
        .btn-custom:hover { background-color: #2ea043; color: white; }
        .terminal-box { background-color: #010409; border: 1px solid #30363d; border-radius: 6px; padding: 15px; font-family: 'Courier New', monospace; color: #58a6ff; max-height: 250px; overflow-y: auto; }
        .nav-tabs .nav-link { color: #8b949e; border: none; }
        .nav-tabs .nav-link.active { background-color: #161b22; color: #58a6ff; border-bottom: 2px solid #58a6ff; }
    </style>
</head>
<body>

<nav class="navbar navbar-expand-lg navbar-dark px-4">
    <a class="navbar-brand fw-bold text-primary" href="/">
        <i class="fa-solid fa-shield-halved me-2"></i>AI Fair Play Detection Portal
    </a>
    <div class="ms-auto d-flex align-items-center">
        <span class="badge bg-success me-3"><i class="fa-solid fa-circle-dot me-1"></i> System Online</span>
        <a href="/dashboard/" class="btn btn-outline-info btn-sm">
            <i class="fa-solid fa-chart-line me-1"></i> Open Analytics Dashboard
        </a>
    </div>
</nav>

<div class="container-fluid py-4 px-4">
    <div class="row g-3 mb-4">
        <div class="col-md-3">
            <div class="card stat-card p-3">
                <div class="text-muted small">Total Player Activities</div>
                <div class="stat-val">{{ stats.players }}</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card warning p-3">
                <div class="text-muted small">Total Alerts Triggered</div>
                <div class="stat-val">{{ stats.alerts }}</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card danger p-3">
                <div class="text-muted small">High-Severity Cheater Actions</div>
                <div class="stat-val text-danger">{{ stats.high_alerts }}</div>
            </div>
        </div>
        <div class="col-md-3">
            <div class="card stat-card success p-3">
                <div class="text-muted small">File Integrity Records</div>
                <div class="stat-val text-success">{{ stats.integrity }}</div>
            </div>
        </div>
    </div>

    <ul class="nav nav-tabs mb-4" id="mainTabs" role="tablist">
        <li class="nav-item"><button class="nav-link active" data-bs-toggle="tab" data-bs-target="#overview"><i class="fa-solid fa-house me-1"></i> Overview & Architecture</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#sim"><i class="fa-solid fa-gamepad me-1"></i> Live Detection Simulation</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#integrity"><i class="fa-solid fa-file-shield me-1"></i> File Integrity Scanner</button></li>
        <li class="nav-item"><button class="nav-link" data-bs-toggle="tab" data-bs-target="#xai"><i class="fa-solid fa-brain me-1"></i> Explainable AI (SHAP)</button></li>
    </ul>

    <div class="tab-content" id="mainTabContent">
        <div class="tab-pane fade show active" id="overview">
            <div class="row">
                <div class="col-md-7">
                    <div class="card">
                        <div class="card-header"><i class="fa-solid fa-circle-info me-2"></i>System Mission & Scope</div>
                        <div class="card-body">
                            <p>Online gaming and esports require real-time, adaptive fair play enforcement. This system integrates <strong>Isolation Forest</strong> for anomaly detection, <strong>Random Forest</strong> for multi-feature cheat classification, <strong>SHA-256 Hashing</strong> for file verification, and <strong>SHAP</strong> for Explainable AI.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-5">
                    <div class="card">
                        <div class="card-header"><i class="fa-solid fa-bell me-2"></i>Live Alert Feed</div>
                        <div class="card-body p-0">
                            <div class="table-responsive">
                                <table class="table table-dark table-hover mb-0">
                                    <thead>
                                        <tr class="text-muted small"><th>Time</th><th>Player</th><th>Severity</th><th>Action</th></tr>
                                    </thead>
                                    <tbody>
                                        {% for a in alerts %}
                                        <tr>
                                            <td class="small">{{ a.timestamp.split(' ')[1] if ' ' in a.timestamp else a.timestamp }}</td>
                                            <td>#{{ a.player_id }}</td>
                                            <td><span class="badge badge-{{ a.severity }}">{{ a.severity }}</span></td>
                                            <td class="small">{{ a.action }}</td>
                                        </tr>
                                        {% else %}
                                        <tr><td colspan="4" class="text-center text-muted py-3">No alerts registered yet</td></tr>
                                        {% endfor %}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="sim">
            <div class="row">
                <div class="col-md-5">
                    <div class="card">
                        <div class="card-header"><i class="fa-solid fa-play me-2"></i>Trigger Telemetry Monitor</div>
                        <div class="card-body">
                            <button class="btn btn-custom w-100 mb-3" onclick="runSimulation()"><i class="fa-solid fa-satellite-dish me-2"></i> Run Live Monitoring Stream</button>
                        </div>
                    </div>
                </div>
                <div class="col-md-7">
                    <div class="card">
                        <div class="card-header"><i class="fa-solid fa-terminal me-2"></i>Stream Log</div>
                        <div class="card-body"><div class="terminal-box" id="sim-terminal">Ready for telemetry stream...</div></div>
                    </div>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="integrity">
            <div class="card">
                <div class="card-header"><i class="fa-solid fa-fingerprint me-2"></i>SHA-256 Game File Verification</div>
                <div class="card-body">
                    <button class="btn btn-primary mb-3" onclick="runIntegrityCheck()"><i class="fa-solid fa-shield-virus me-2"></i> Execute File Integrity Check</button>
                    <div class="terminal-box" id="integrity-terminal">Click above to scan game files.</div>
                </div>
            </div>
        </div>

        <div class="tab-pane fade" id="xai">
            <div class="card">
                <div class="card-header"><i class="fa-solid fa-lightbulb me-2"></i>SHAP Decision Reasoning</div>
                <div class="card-body">
                    <button class="btn btn-info text-dark mb-3" onclick="runXAI()"><i class="fa-solid fa-calculator me-2"></i> Generate SHAP Explanation</button>
                    <div class="terminal-box" id="xai-terminal">Click to explain a flagged decision.</div>
                </div>
            </div>
        </div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
function runSimulation() {
    document.getElementById('sim-terminal').innerText = "Streaming telemetry...";
    fetch('/api/run-monitoring').then(res => res.json()).then(data => { document.getElementById('sim-terminal').innerText = data.output; });
}
function runIntegrityCheck() {
    document.getElementById('integrity-terminal').innerText = "Hashing game files...";
    fetch('/api/run-integrity').then(res => res.json()).then(data => { document.getElementById('integrity-terminal').innerText = data.output; });
}
function runXAI() {
    document.getElementById('xai-terminal').innerText = "Computing SHAP values...";
    fetch('/api/run-xai').then(res => res.json()).then(data => { document.getElementById('xai-terminal').innerText = data.output; });
}
</script>
</body>
</html>
"""

# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template_string(MAIN_HTML, stats=get_db_stats(), alerts=get_recent_alerts())

@app.route("/api/run-monitoring")
def api_run_monitoring():
    try:
        script = os.path.join(PROJECT_ROOT, "src", "monitoring", "real_time_monitor.py")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=PROJECT_ROOT)
        db_script = os.path.join(PROJECT_ROOT, "src", "database", "database_manager.py")
        subprocess.run([sys.executable, db_script], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return jsonify({"output": res.stdout + ("\n[STDERR]\n" + res.stderr if res.stderr else "")})
    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})

@app.route("/api/run-integrity")
def api_run_integrity():
    try:
        script = os.path.join(PROJECT_ROOT, "src", "file_integrity", "file_integrity_checker.py")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return jsonify({"output": res.stdout + ("\n[STDERR]\n" + res.stderr if res.stderr else "")})
    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})

@app.route("/api/run-xai")
def api_run_xai():
    try:
        script = os.path.join(PROJECT_ROOT, "src", "explainable_ai", "explain_player_prediction.py")
        res = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=PROJECT_ROOT)
        return jsonify({"output": res.stdout + ("\n[STDERR]\n" + res.stderr if res.stderr else "")})
    except Exception as e:
        return jsonify({"output": f"Error: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)