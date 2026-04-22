"""
CineBandit — Streamlit Frontend
Three screens:
  1. 🎬 Recommendations  — get movie recommendations, submit feedback
  2. 🔧 Pipeline Console — Airflow DAG status, MLflow model info, drift score
  3. 📊 Monitoring       — live metrics charts from Prometheus/API
"""

import os
import time
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

API_URL     = os.environ.get("API_URL", "http://api:8000")
AIRFLOW_URL = os.environ.get("AIRFLOW_URL", "http://airflow-webserver:8080")
MLFLOW_URL  = os.environ.get("MLFLOW_URL", "http://mlflow:5000")
PROM_URL    = os.environ.get("PROM_URL", "http://prometheus:9090")

# Genre emoji map for visual flair
GENRE_EMOJI = {
    "Action": "💥", "Adventure": "🗺️", "Animation": "🎨",
    "Children's": "🧸", "Comedy": "😂", "Crime": "🔍",
    "Documentary": "📽️", "Drama": "🎭", "Fantasy": "🧙",
    "Film-Noir": "🕵️", "Horror": "👻", "Musical": "🎵",
    "Mystery": "🔮", "Romance": "❤️", "Sci-Fi": "🚀",
    "Thriller": "😰", "War": "⚔️", "Western": "🤠", "unknown": "🎬"
}

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CineBandit",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --bg: #0a0a0f;
        --surface: #13131a;
        --surface2: #1c1c28;
        --border: #2a2a3d;
        --accent: #e8c547;
        --accent2: #7c6af5;
        --text: #e8e8f0;
        --text-muted: #8888aa;
        --success: #4ade80;
        --danger: #f87171;
        --warning: #fbbf24;
    }

    /* Global */
    .stApp {
        background-color: var(--bg);
        font-family: 'DM Sans', sans-serif;
        color: var(--text);
    }

    # /* Hide Streamlit branding */
    # #MainMenu, footer{ visibility: hidden; }
    # .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] .stMarkdown h1 {
        font-family: 'DM Serif Display', serif;
        color: var(--accent);
        font-size: 1.8rem;
        letter-spacing: -0.02em;
    }

    /* Metric cards */
    .metric-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
    }
    .metric-label {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: var(--text);
        line-height: 1;
    }
    .metric-delta {
        font-size: 0.8rem;
        color: var(--success);
        margin-top: 0.2rem;
    }

    /* Movie cards */
    .movie-card {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 0.8rem;
        transition: border-color 0.2s;
        position: relative;
    }
    .movie-card:hover { border-color: var(--accent2); }
    .movie-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.15rem;
        color: var(--text);
        margin-bottom: 0.3rem;
    }
    .movie-meta {
        font-size: 0.78rem;
        color: var(--text-muted);
        margin-bottom: 0.5rem;
    }
    .genre-tag {
        display: inline-block;
        background: rgba(124, 106, 245, 0.15);
        border: 1px solid rgba(124, 106, 245, 0.3);
        color: #a89cf7;
        font-size: 0.7rem;
        font-weight: 500;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .ucb-badge {
        position: absolute;
        top: 1rem;
        right: 1rem;
        background: rgba(232, 197, 71, 0.1);
        border: 1px solid rgba(232, 197, 71, 0.3);
        color: var(--accent);
        font-size: 0.7rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 20px;
    }

    /* Status badges */
    .status-ok {
        background: rgba(74, 222, 128, 0.1);
        border: 1px solid rgba(74, 222, 128, 0.3);
        color: var(--success);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-error {
        background: rgba(248, 113, 113, 0.1);
        border: 1px solid rgba(248, 113, 113, 0.3);
        color: var(--danger);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .status-warn {
        background: rgba(251, 191, 36, 0.1);
        border: 1px solid rgba(251, 191, 36, 0.3);
        color: var(--warning);
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Section headers */
    .section-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: var(--text);
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    .section-sub {
        font-size: 0.82rem;
        color: var(--text-muted);
        margin-bottom: 1.5rem;
    }

    /* DAG status table */
    .dag-row {
        display: flex;
        align-items: center;
        padding: 0.7rem 1rem;
        border-bottom: 1px solid var(--border);
        gap: 1rem;
    }
    .dag-name {
        font-weight: 500;
        font-size: 0.88rem;
        flex: 1;
        color: var(--text);
    }
    .dag-schedule {
        font-size: 0.75rem;
        color: var(--text-muted);
        flex: 1;
    }

    /* Streamlit overrides */
    .stButton > button {
        background: var(--accent2);
        color: white;
        border: none;
        border-radius: 8px;
        font-family: 'DM Sans', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .stSelectbox > div > div {
        background: var(--surface2);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
    }
    .stSlider > div { color: var(--text); }

    /* Feedback buttons */
    div[data-testid="column"] .stButton > button {
        width: 100%;
    }

    /* Alert box */
    .drift-alert {
        background: rgba(248, 113, 113, 0.08);
        border: 1px solid rgba(248, 113, 113, 0.4);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        color: var(--danger);
        font-size: 0.85rem;
        font-weight: 500;
    }
    .drift-ok {
        background: rgba(74, 222, 128, 0.06);
        border: 1px solid rgba(74, 222, 128, 0.3);
        border-radius: 10px;
        padding: 0.8rem 1.2rem;
        color: var(--success);
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ── API helpers ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=5)
def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


@st.cache_data(ttl=30)
def get_mlflow_runs() -> list:
    try:
        r = requests.post(
            f"{MLFLOW_URL}/api/2.0/mlflow/runs/search",
            json={"experiment_ids": ["1"], "max_results": 10,
                  "order_by": ["start_time DESC"]},
            timeout=5
        )
        return r.json().get("runs", [])
    except Exception:
        return []


@st.cache_data(ttl=30)
def get_airflow_dags() -> list:
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags",
            auth=("admin", "admin"),
            timeout=5
        )
        return r.json().get("dags", [])
    except Exception:
        return []


@st.cache_data(ttl=15)
def get_airflow_dag_runs(dag_id: str) -> list:
    try:
        r = requests.get(
            f"{AIRFLOW_URL}/api/v1/dags/{dag_id}/dagRuns?limit=5&order_by=-execution_date",
            auth=("admin", "admin"),
            timeout=5
        )
        return r.json().get("dag_runs", [])
    except Exception:
        return []


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🎬 CineBandit")
    st.markdown("*Adaptive Movie Recommendations*")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🎬 Recommendations", "🔧 Pipeline Console", "📊 Monitoring"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # API health indicator
    health = api_get("/health")
    ready  = api_get("/ready")

    if health and health.get("status") == "ok":
        st.markdown('<span class="status-ok">● API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-error">● API Offline</span>', unsafe_allow_html=True)

    if ready and ready.get("model_loaded"):
        ts = ready.get("model_timestep", 0)
        st.markdown(f'<span class="status-ok">● Model Loaded (t={ts:,})</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-warn">● Model Not Loaded</span>',
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        '<span style="font-size:0.72rem;color:#8888aa;">Built with LinUCB · MLflow · Airflow</span>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

if page == "🎬 Recommendations":

    st.markdown('<div class="section-title">Movie Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">The LinUCB bandit learns your preferences in real time</div>',
                unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.markdown("**User Profile**")
        user_id = st.selectbox(
            "Select User",
            options=list(range(1, 944)),
            format_func=lambda x: f"User {x}",
            label_visibility="collapsed"
        )
        n_recs = st.slider("Recommendations", 1, 10, 5)

        get_btn = st.button("🎲 Get Recommendations", width='stretch')

        st.markdown("---")

        # Snapshot metrics
        snapshot = api_get("/snapshot")
        if snapshot:
            st.markdown('<div class="metric-card">'
                        '<div class="metric-label">Rolling CTR</div>'
                        f'<div class="metric-value">{snapshot["rolling_ctr"]:.1%}</div>'
                        '<div class="metric-delta">last 100 feedbacks</div>'
                        '</div>', unsafe_allow_html=True)

            st.markdown('<div class="metric-card">'
                        '<div class="metric-label">Model Updates</div>'
                        f'<div class="metric-value">{snapshot["model_timestep"]:,}</div>'
                        '<div class="metric-delta">incremental updates</div>'
                        '</div>', unsafe_allow_html=True)

            drift = snapshot["drift_score"]
            drift_color = "#f87171" if drift > 0.3 else "#4ade80"
            st.markdown(f'<div class="metric-card">'
                        f'<div class="metric-label">Drift Score</div>'
                        f'<div class="metric-value" style="color:{drift_color}">{drift:.3f}</div>'
                        f'<div class="metric-delta">KL divergence (threshold: 0.3)</div>'
                        f'</div>', unsafe_allow_html=True)

    with col_right:

        # Session state for recommendations
        if "recommendations" not in st.session_state:
            st.session_state.recommendations = []
        if "feedback_sent" not in st.session_state:
            st.session_state.feedback_sent = {}
        if "session_ctr_history" not in st.session_state:
            st.session_state.session_ctr_history = []

        if get_btn:
            with st.spinner("Asking the bandit..."):
                result = api_post("/recommend", {"user_id": user_id, "n": n_recs})
            if result:
                st.session_state.recommendations = result.get("recommendations", [])
                st.session_state.feedback_sent = {}

        if st.session_state.recommendations:
            st.markdown(f"**Top {len(st.session_state.recommendations)} picks for User {user_id}**")

            for rec in st.session_state.recommendations:
                mid   = rec["movie_id"]
                sent  = st.session_state.feedback_sent.get(mid)

                genres_html = "".join(
                    f'<span class="genre-tag">'
                    f'{GENRE_EMOJI.get(g, "🎬")} {g}'
                    f'</span>'
                    for g in rec["genres"]
                )

                st.markdown(
                    f'<div class="movie-card">'
                    f'<span class="ucb-badge">UCB {rec["ucb_score"]:.3f}</span>'
                    f'<div class="movie-title">{rec["title"]}</div>'
                    f'<div class="movie-meta">{rec["year"]}</div>'
                    f'<div>{genres_html}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if sent:
                    emoji = {"like": "👍", "dislike": "👎", "skip": "⏭️"}[sent]
                    st.markdown(
                        f'<div style="font-size:0.8rem;color:#8888aa;'
                        f'margin:-0.5rem 0 0.8rem 0.5rem;">'
                        f'{emoji} Feedback sent</div>',
                        unsafe_allow_html=True
                    )
                else:
                    fc1, fc2, fc3 = st.columns(3)
                    with fc1:
                        if st.button("👍 Like", key=f"like_{mid}"):
                            r = api_post("/feedback", {
                                "user_id": user_id, "movie_id": mid, "reaction": "like"
                            })
                            if r:
                                st.session_state.feedback_sent[mid] = "like"
                                st.session_state.session_ctr_history.append(1)
                                st.rerun()
                    with fc2:
                        if st.button("👎 Dislike", key=f"dislike_{mid}"):
                            r = api_post("/feedback", {
                                "user_id": user_id, "movie_id": mid, "reaction": "dislike"
                            })
                            if r:
                                st.session_state.feedback_sent[mid] = "dislike"
                                st.session_state.session_ctr_history.append(0)
                                st.rerun()
                    with fc3:
                        if st.button("⏭️ Skip", key=f"skip_{mid}"):
                            r = api_post("/feedback", {
                                "user_id": user_id, "movie_id": mid, "reaction": "skip"
                            })
                            if r:
                                st.session_state.feedback_sent[mid] = "skip"
                                st.session_state.session_ctr_history.append(0)
                                st.rerun()

            # Session CTR mini-chart
            if len(st.session_state.session_ctr_history) >= 2:
                st.markdown("---")
                st.markdown("**Session CTR**")
                ctr_series = pd.Series(st.session_state.session_ctr_history).expanding().mean()
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=ctr_series, mode="lines+markers",
                    line=dict(color="#7c6af5", width=2),
                    marker=dict(size=5)
                ))
                fig.update_layout(
                    height=150, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, showticklabels=False),
                    yaxis=dict(showgrid=True, gridcolor="#2a2a3d",
                               range=[0, 1], tickformat=".0%",
                               tickfont=dict(color="#8888aa", size=10)),
                    font=dict(color="#8888aa")
                )
                st.plotly_chart(fig, width='stretch')

        else:
            st.markdown(
                '<div style="text-align:center;padding:4rem 2rem;'
                'color:#8888aa;font-size:0.9rem;">'
                '🎬 Select a user and click <strong>Get Recommendations</strong>'
                '</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 2 — PIPELINE CONSOLE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔧 Pipeline Console":

    st.markdown('<div class="section-title">Pipeline Console</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Airflow DAG status · MLflow model registry · Drift monitoring</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        # ── Airflow DAGs ──────────────────────────────────────────────────────
        st.markdown("#### Airflow Pipelines")

        dag_configs = [
            ("cinebandit_ingestion",       "Data Ingestion",    "@daily"),
            ("cinebandit_drift_detection", "Drift Detection",   "Every 6h"),
            ("cinebandit_retraining",      "Model Retraining",  "On-demand"),
        ]

        for dag_id, display_name, schedule in dag_configs:
            runs = get_airflow_dag_runs(dag_id)
            if runs:
                last = runs[0]
                state = last.get("state", "unknown")
                last_run = last.get("execution_date", "")[:16].replace("T", " ")
                if state == "success":
                    badge = '<span class="status-ok">✓ Success</span>'
                elif state == "running":
                    badge = '<span class="status-warn">⟳ Running</span>'
                elif state == "failed":
                    badge = '<span class="status-error">✗ Failed</span>'
                else:
                    badge = f'<span class="status-warn">{state}</span>'
            else:
                badge = '<span class="status-warn">No runs yet</span>'
                last_run = "—"

            st.markdown(
                f'<div class="dag-row">'
                f'<div class="dag-name">{display_name}</div>'
                f'<div class="dag-schedule">{schedule}</div>'
                f'<div style="font-size:0.72rem;color:#8888aa;min-width:110px">{last_run}</div>'
                f'<div>{badge}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # ── Manual triggers ───────────────────────────────────────────────────
        st.markdown("#### Manual Triggers")
        tc1, tc2, tc3 = st.columns(3)

        with tc1:
            if st.button("▶ Run Ingestion", width='stretch'):
                try:
                    r = requests.post(
                        f"{AIRFLOW_URL}/api/v1/dags/cinebandit_ingestion/dagRuns",
                        json={"conf": {}},
                        auth=("admin", "admin"),
                        timeout=5
                    )
                    if r.status_code in [200, 201]:
                        st.success("Ingestion DAG triggered")
                    else:
                        st.error(f"Failed: {r.status_code}")
                except Exception as e:
                    st.error(str(e))

        with tc2:
            if st.button("▶ Run Drift Check", width='stretch'):
                try:
                    r = requests.post(
                        f"{AIRFLOW_URL}/api/v1/dags/cinebandit_drift_detection/dagRuns",
                        json={"conf": {}},
                        auth=("admin", "admin"),
                        timeout=5
                    )
                    if r.status_code in [200, 201]:
                        st.success("Drift detection triggered")
                    else:
                        st.error(f"Failed: {r.status_code}")
                except Exception as e:
                    st.error(str(e))

        with tc3:
            if st.button("▶ Retrain Model", width='stretch'):
                try:
                    r = requests.post(
                        f"{AIRFLOW_URL}/api/v1/dags/cinebandit_retraining/dagRuns",
                        json={"conf": {"reason": "manual_ui"}},
                        auth=("admin", "admin"),   # 🔥 ADD THIS
                        timeout=5
                    )
                    if r.status_code in [200, 201]:
                        st.success("Retraining triggered")
                    else:
                        st.error(f"Failed: {r.status_code}")
                except Exception as e:
                    st.error(str(e))

        st.markdown("---")

        # ── MLflow recent runs ────────────────────────────────────────────────
        st.markdown("#### Recent MLflow Runs")
        runs = get_mlflow_runs()

        if runs:
            run_data = []
            for run in runs[:8]:
                info = run.get("info", {})

                params_list  = run.get("data", {}).get("params", [])
                metrics_list = run.get("data", {}).get("metrics", [])

                params  = {p["key"]: p["value"] for p in params_list}
                metrics = {m["key"]: m["value"] for m in metrics_list}

                run_data.append({
                    "Run":       info.get("run_name", info.get("run_id", "")[:8]),
                    "Alpha":     params.get("alpha", "—"),
                    "Train CTR": f"{float(metrics['train_ctr']):.4f}"
                                if "train_ctr" in metrics else "—",
                    "Test CTR":  f"{float(metrics['test_ctr']):.4f}"
                                if "test_ctr" in metrics else "—",
                    "Status":    "✓" if info.get("status") == "FINISHED" else "⏳",
                })
            st.dataframe(
                pd.DataFrame(run_data),
                hide_index=True,
                width='stretch'
            )
        else:
            st.info("No MLflow runs found. Run `make train-mlflow` first.")

    with col2:
        # ── Model info ────────────────────────────────────────────────────────
        st.markdown("#### Active Model")
        model_info = api_get("/model/info")

        if model_info:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Algorithm</div>'
                f'<div class="metric-value" style="font-size:1.3rem">{model_info["algorithm"]}</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Exploration (α)</div>'
                f'<div class="metric-value">{model_info["alpha"]}</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Total Updates</div>'
                f'<div class="metric-value">{model_info["timesteps"]:,}</div>'
                f'</div>', unsafe_allow_html=True
            )
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Movies Seen</div>'
                f'<div class="metric-value">{model_info["arms_seen"]:,}'
                f'<span style="font-size:1rem;color:#8888aa"> / {model_info["total_movies"]}</span>'
                f'</div>'
                f'</div>', unsafe_allow_html=True
            )
        else:
            st.warning("Model not loaded")

        st.markdown("---")

        # ── Drift status ──────────────────────────────────────────────────────
        st.markdown("#### Drift Status")
        drift_status = api_get("/drift/status")

        if drift_status:
            score = drift_status["drift_score"]
            threshold = drift_status["threshold"]

            # Gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"font": {"color": "#e8e8f0", "size": 32}},
                gauge={
                    "axis": {"range": [0, 0.6],
                             "tickcolor": "#8888aa",
                             "tickfont": {"color": "#8888aa", "size": 10}},
                    "bar": {"color": "#f87171" if score > threshold else "#4ade80",
                            "thickness": 0.6},
                    "bgcolor": "#1c1c28",
                    "bordercolor": "#2a2a3d",
                    "steps": [
                        {"range": [0, threshold], "color": "rgba(74,222,128,0.08)"},
                        {"range": [threshold, 0.6], "color": "rgba(248,113,113,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 2},
                        "thickness": 0.8,
                        "value": threshold
                    }
                }
            ))
            fig.update_layout(
                height=200,
                margin=dict(l=20, r=20, t=20, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e8e8f0")
            )
            st.plotly_chart(fig, width='stretch')

            if drift_status["alert"]:
                st.markdown(
                    '<div class="drift-alert">🚨 Drift detected! '
                    'Retraining recommended.</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="drift-ok">✅ Distribution stable. '
                    'No retraining needed.</div>',
                    unsafe_allow_html=True
                )

        # ── Reload model button ───────────────────────────────────────────────
        st.markdown("---")
        if st.button("🔄 Reload Model", width='stretch'):
            r = api_post("/model/reload", {})
            if r and r.get("status") == "ok":
                st.success(f"Model reloaded (t={r.get('timesteps', '?'):,})")
                st.cache_data.clear()


# ══════════════════════════════════════════════════════════════════════════════
# SCREEN 3 — MONITORING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Monitoring":

    st.markdown('<div class="section-title">Live Monitoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Real-time metrics from the CineBandit API</div>',
                unsafe_allow_html=True)

    # Auto-refresh toggle
    auto_refresh = st.toggle("Auto-refresh (10s)", value=False)
    if auto_refresh:
        time.sleep(10)
        st.rerun()

    # ── Top metrics row ───────────────────────────────────────────────────────
    snapshot = api_get("/snapshot")
    model_info = api_get("/model/info")

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        ctr = snapshot["rolling_ctr"] if snapshot else 0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Rolling CTR</div>'
            f'<div class="metric-value">{ctr:.1%}</div>'
            f'<div class="metric-delta">last 100 events</div>'
            f'</div>', unsafe_allow_html=True
        )

    with m2:
        ts = snapshot["model_timestep"] if snapshot else 0
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Model Timestep</div>'
            f'<div class="metric-value">{ts:,}</div>'
            f'<div class="metric-delta">incremental updates</div>'
            f'</div>', unsafe_allow_html=True
        )

    with m3:
        arms = snapshot["arms_seen"] if snapshot else 0
        total = model_info["total_movies"] if model_info else 1682
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Arm Coverage</div>'
            f'<div class="metric-value">{arms/total:.1%}</div>'
            f'<div class="metric-delta">{arms:,} / {total:,} movies</div>'
            f'</div>', unsafe_allow_html=True
        )

    with m4:
        drift = snapshot["drift_score"] if snapshot else 0
        color = "#f87171" if drift > 0.3 else "#4ade80"
        st.markdown(
            f'<div class="metric-card">'
            f'<div class="metric-label">Drift Score</div>'
            f'<div class="metric-value" style="color:{color}">{drift:.3f}</div>'
            f'<div class="metric-delta">KL divergence</div>'
            f'</div>', unsafe_allow_html=True
        )

    st.markdown("---")

    # ── Prometheus raw metrics ────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Request Rate")
        try:
            raw = requests.get(f"{API_URL}/metrics", timeout=5).text
            req_total = 0
            feedback_likes = 0
            feedback_dislikes = 0
            feedback_skips = 0

            for line in raw.split("\n"):
                if line.startswith("cinebandit_recommendation_requests_total{status=\"success\"}"):
                    try:
                        req_total = int(float(line.split()[-1]))
                    except Exception:
                        pass
                if "cinebandit_feedback_total{reaction=\"like\"}" in line and not line.startswith("#"):
                    try:
                        feedback_likes = int(float(line.split()[-1]))
                    except Exception:
                        pass
                if "cinebandit_feedback_total{reaction=\"dislike\"}" in line and not line.startswith("#"):
                    try:
                        feedback_dislikes = int(float(line.split()[-1]))
                    except Exception:
                        pass
                if "cinebandit_feedback_total{reaction=\"skip\"}" in line and not line.startswith("#"):
                    try:
                        feedback_skips = int(float(line.split()[-1]))
                    except Exception:
                        pass

            # Feedback breakdown pie
            labels = ["Likes", "Dislikes", "Skips"]
            values = [feedback_likes, feedback_dislikes, feedback_skips]

            if sum(values) > 0:
                fig = go.Figure(go.Pie(
                    labels=labels, values=values,
                    hole=0.55,
                    marker=dict(colors=["#4ade80", "#f87171", "#8888aa"]),
                    textfont=dict(color="#e8e8f0", size=12),
                ))
                fig.update_layout(
                    height=280,
                    margin=dict(l=0, r=0, t=10, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    showlegend=True,
                    legend=dict(font=dict(color="#8888aa", size=11)),
                    annotations=[dict(
                        text=f"<b>{sum(values)}</b><br>total",
                        x=0.5, y=0.5, font_size=14,
                        font_color="#e8e8f0",
                        showarrow=False
                    )]
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No feedback data yet. Try making recommendations on Screen 1.")

            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">Total Recommendation Requests</div>'
                f'<div class="metric-value">{req_total:,}</div>'
                f'</div>', unsafe_allow_html=True
            )

        except Exception as e:
            st.warning(f"Could not fetch Prometheus metrics: {e}")

    with col_b:
        st.markdown("#### Service Health")

        services = [
            ("API",        f"{API_URL}/health",   None,            None),
            ("MLflow",     f"{MLFLOW_URL}/health", None,           None),
            ("Airflow",    f"{AIRFLOW_URL}/health", None,          None),
            ("Prometheus", f"{PROM_URL}/-/ready", None, None),
        ]

        for name, url, user, pwd in services:
            try:
                kwargs = {"timeout": 3}
                if user:
                    kwargs["auth"] = (user, pwd)
                r = requests.get(url, **kwargs)
                if r.status_code < 400:
                    badge = '<span class="status-ok">● Online</span>'
                else:
                    badge = f'<span class="status-error">● {r.status_code}</span>'
            except Exception:
                badge = '<span class="status-error">● Offline</span>'

            st.markdown(
                f'<div class="dag-row">'
                f'<div class="dag-name">{name}</div>'
                f'<div>{badge}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("#### Grafana Dashboard")
        st.markdown(
            "Full time-series monitoring is available in Grafana:",
        )
        st.markdown(
            "**[Open Grafana →](http://localhost:3000)**  "
            "*(admin / admin)*"
        )
        st.markdown(
            "The CineBandit dashboard shows:\n"
            "- Recommendation latency (p50/p95/p99)\n"
            "- Feedback breakdown over time\n"
            "- Cumulative reward curve\n"
            "- Drift score history\n"
            "- Model alpha and timestep"
        )

    # ── Raw Prometheus metrics expander ───────────────────────────────────────
    with st.expander("🔍 Raw Prometheus Metrics"):
        try:
            raw = requests.get(f"{API_URL}/metrics", timeout=5).text
            cinebandit_lines = [
                l for l in raw.split("\n")
                if "cinebandit" in l and not l.startswith("#")
            ]
            st.code("\n".join(cinebandit_lines) or "No metrics yet", language="text")
        except Exception:
            st.error("Could not fetch /metrics endpoint")