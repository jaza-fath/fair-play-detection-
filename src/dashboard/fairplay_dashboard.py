import os
import sqlite3
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output
import dash_bootstrap_components as dbc


DB_PATH = "data/database/fairplay.db"
REPORT_DIR = "reports"
REPORT_PATH = os.path.join(REPORT_DIR, "fairplay_report.txt")


# -----------------------------
# Database helpers
# -----------------------------
def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn, table_name):
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,)
    )
    return cursor.fetchone() is not None


def safe_read_table(conn, table_name, limit=None):
    if not table_exists(conn, table_name):
        return pd.DataFrame()

    query = f"SELECT * FROM {table_name} ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    return pd.read_sql_query(query, conn)


def load_dashboard_data():
    if not os.path.exists(DB_PATH):
        return {
            "summary": {
                "player_activity": 0,
                "alerts": 0,
                "file_integrity_records": 0,
                "high_severity_alerts": 0
            },
            "player_df": pd.DataFrame(),
            "alert_df": pd.DataFrame(),
            "integrity_df": pd.DataFrame()
        }

    with get_connection() as conn:
        player_df = safe_read_table(conn, "player_activity", limit=100)
        alert_df = safe_read_table(conn, "alerts", limit=100)
        integrity_df = safe_read_table(conn, "file_integrity_records", limit=100)

        summary = {
            "player_activity": len(player_df),
            "alerts": len(alert_df),
            "file_integrity_records": len(integrity_df),
            "high_severity_alerts": int(
                (alert_df["severity"] == "HIGH").sum()
            ) if not alert_df.empty and "severity" in alert_df.columns else 0
        }

    return {
        "summary": summary,
        "player_df": player_df,
        "alert_df": alert_df,
        "integrity_df": integrity_df
    }


# -----------------------------
# Figure helpers
# -----------------------------
def blank_figure(title):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text": "No data available",
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 16}
            }
        ]
    )
    return fig


def severity_figure(alert_df):
    if alert_df.empty or "severity" not in alert_df.columns:
        return blank_figure("Alert Severity Distribution")

    counts = alert_df["severity"].fillna("UNKNOWN").value_counts()

    fig = go.Figure(
        data=[
            go.Pie(
                labels=counts.index.tolist(),
                values=counts.values.tolist(),
                hole=0.45
            )
        ]
    )
    fig.update_layout(
        title="Alert Severity Distribution",
        template="plotly_white"
    )
    return fig


def action_figure(player_df):
    if player_df.empty or "action" not in player_df.columns:
        return blank_figure("Recommended Action Distribution")

    counts = player_df["action"].fillna("UNKNOWN").value_counts()

    fig = go.Figure(
        data=[
            go.Bar(
                x=counts.index.tolist(),
                y=counts.values.tolist(),
                marker_color="#1f77b4"
            )
        ]
    )
    fig.update_layout(
        title="Recommended Action Distribution",
        template="plotly_white",
        xaxis_title="Action",
        yaxis_title="Count"
    )
    return fig


def risk_figure(player_df):
    if player_df.empty or "risk_score" not in player_df.columns:
        return blank_figure("Risk Score Distribution")

    fig = go.Figure(
        data=[
            go.Histogram(
                x=player_df["risk_score"],
                nbinsx=20,
                marker_color="#ff7f0e"
            )
        ]
    )
    fig.update_layout(
        title="Risk Score Distribution",
        template="plotly_white",
        xaxis_title="Risk Score",
        yaxis_title="Frequency"
    )
    return fig


# -----------------------------
# Reporting
# -----------------------------
def generate_report(summary, alert_df, player_df):
    os.makedirs(REPORT_DIR, exist_ok=True)

    lines = []
    lines.append("=" * 60)
    lines.append("FAIR PLAY DETECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated At : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("SUMMARY")
    lines.append("-" * 60)
    lines.append(f"Player Activity Records : {summary['player_activity']}")
    lines.append(f"Alert Records           : {summary['alerts']}")
    lines.append(f"Integrity Records       : {summary['file_integrity_records']}")
    lines.append(f"High Severity Alerts    : {summary['high_severity_alerts']}")
    lines.append("")

    lines.append("RECENT ALERTS")
    lines.append("-" * 60)

    if alert_df.empty:
        lines.append("No alerts found.")
    else:
        for _, row in alert_df.head(10).iterrows():
            lines.append(
                f"[{row.get('timestamp', '')}] "
                f"Player {row.get('player_id', '')} | "
                f"Match {row.get('match_id', '')} | "
                f"Severity: {row.get('severity', '')} | "
                f"Action: {row.get('action', '')}"
            )
            if "message" in row:
                lines.append(f"  Message: {row.get('message', '')}")

    lines.append("")
    lines.append("TOP RISK PLAYERS")
    lines.append("-" * 60)

    if player_df.empty or "risk_score" not in player_df.columns:
        lines.append("No player activity found.")
    else:
        top_risk = player_df.sort_values(by="risk_score", ascending=False).head(10)
        for _, row in top_risk.iterrows():
            lines.append(
                f"[{row.get('timestamp', '')}] "
                f"Player {row.get('player_id', '')} | "
                f"Risk: {row.get('risk_score', '')} | "
                f"Severity: {row.get('severity', '')} | "
                f"Action: {row.get('action', '')}"
            )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return REPORT_PATH


# -----------------------------
# Dash app
# -----------------------------
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Fair Play Detection Dashboard"
server = app.server


def make_kpi_card(title, value, color):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="card-title"),
                html.H3(value, className="card-text"),
            ]
        ),
        color=color,
        inverse=True,
        className="shadow-sm",
    )


app.layout = dbc.Container(
    [
        html.H1("AI-Based Fair Play Detection Dashboard", className="mt-4"),
        html.P(
            "Monitor player activity, alerts, and cheat risk in real time.",
            className="text-muted"
        ),

        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),

        html.Div(id="last-updated", className="mb-3"),

        dbc.Row(
            [
                dbc.Col(html.Div(id="kpi-players"), md=3),
                dbc.Col(html.Div(id="kpi-alerts"), md=3),
                dbc.Col(html.Div(id="kpi-integrity"), md=3),
                dbc.Col(html.Div(id="kpi-high-severity"), md=3),
            ],
            className="mb-4"
        ),

        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="severity-chart"), md=4),
                dbc.Col(dcc.Graph(id="action-chart"), md=4),
                dbc.Col(dcc.Graph(id="risk-chart"), md=4),
            ],
            className="mb-4"
        ),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H4("Recent Alerts"),
                        dash_table.DataTable(
                            id="alerts-table",
                            columns=[
                                {"name": "Timestamp", "id": "timestamp"},
                                {"name": "Player ID", "id": "player_id"},
                                {"name": "Match ID", "id": "match_id"},
                                {"name": "Severity", "id": "severity"},
                                {"name": "Action", "id": "action"},
                                {"name": "Message", "id": "message"},
                            ],
                            data=[],
                            page_size=10,
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "8px",
                                "whiteSpace": "normal",
                                "height": "auto",
                            },
                            style_header={
                                "backgroundColor": "#f8f9fa",
                                "fontWeight": "bold"
                            },
                            style_data_conditional=[
                                {
                                    "if": {"filter_query": '{severity} = "HIGH"'},
                                    "backgroundColor": "#f8d7da",
                                    "color": "black",
                                },
                                {
                                    "if": {"filter_query": '{severity} = "MEDIUM"'},
                                    "backgroundColor": "#fff3cd",
                                    "color": "black",
                                },
                            ],
                        ),
                    ],
                    md=12
                )
            ]
        ),

        html.Hr(),
        html.P("Report will be saved automatically in the reports/ folder."),
    ],
    fluid=True
)


@app.callback(
    [
        Output("kpi-players", "children"),
        Output("kpi-alerts", "children"),
        Output("kpi-integrity", "children"),
        Output("kpi-high-severity", "children"),
        Output("severity-chart", "figure"),
        Output("action-chart", "figure"),
        Output("risk-chart", "figure"),
        Output("alerts-table", "data"),
        Output("last-updated", "children"),
    ],
    [Input("refresh-interval", "n_intervals")]
)
def update_dashboard(_):
    data = load_dashboard_data()
    summary = data["summary"]
    player_df = data["player_df"]
    alert_df = data["alert_df"]
    integrity_df = data["integrity_df"]

    # Generate report file
    report_path = generate_report(summary, alert_df, player_df)

    # Prepare table data
    if alert_df.empty:
        alert_records = []
    else:
        cols = [c for c in ["timestamp", "player_id", "match_id", "severity", "action", "message"] if c in alert_df.columns]
        alert_records = alert_df[cols].fillna("").to_dict("records")

    last_updated = html.Div(
        [
            html.Strong("Last Updated: "),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            html.Span(f" | Report saved to: {report_path}", className="text-muted")
        ]
    )

    return (
        make_kpi_card("Player Activity Records", summary["player_activity"], "primary"),
        make_kpi_card("Alert Records", summary["alerts"], "danger"),
        make_kpi_card("Integrity Records", summary["file_integrity_records"], "success"),
        make_kpi_card("High Severity Alerts", summary["high_severity_alerts"], "warning"),
        severity_figure(alert_df),
        action_figure(player_df),
        risk_figure(player_df),
        alert_records,
        last_updated,
    )


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)