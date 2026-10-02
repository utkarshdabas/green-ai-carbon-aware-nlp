import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import pandas as pd
import plotly.express as px
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
)
from codecarbon import EmissionsTracker
import math
import warnings
warnings.filterwarnings("ignore")
torch.set_num_threads(1)

st.set_page_config(page_title="Green AI Dashboard", page_icon="🌍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Exo 2', sans-serif;
}

/* ─── Animated grid background ─── */
.stApp {
    background-color: #020810;
    color: #c8f0e0;
    background-image:
        linear-gradient(rgba(0,255,140,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,140,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    animation: gridpulse 8s ease-in-out infinite;
}

@keyframes gridpulse {
    0%, 100% { background-size: 40px 40px; }
    50%       { background-size: 42px 42px; }
}

/* ─── Scanline overlay ─── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0,0,0,0.07) 2px,
        rgba(0,0,0,0.07) 4px
    );
    pointer-events: none;
    z-index: 0;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background-color: #030d18;
    border-right: 1px solid #00ff8c22;
    box-shadow: 4px 0 40px rgba(0,255,140,0.05);
}

[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
}

/* ─── Headers ─── */
h1, h2, h3 {
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 900;
    letter-spacing: -1px;
    color: #00ff8c !important;
    text-shadow: 0 0 30px rgba(0,255,140,0.4), 0 0 60px rgba(0,255,140,0.15);
    -webkit-text-fill-color: unset !important;
    background: none !important;
}

h1 {
    font-size: 2.8rem !important;
    text-transform: uppercase;
    letter-spacing: 4px;
    animation: flicker 6s infinite;
}

@keyframes flicker {
    0%, 97%, 100% { opacity: 1; }
    98%           { opacity: 0.7; }
    99%           { opacity: 1; }
    99.5%         { opacity: 0.4; }
}

/* ─── Metric Cards ─── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #030d18 0%, #050f1e 100%);
    border: 1px solid #00ff8c33;
    border-radius: 4px;
    padding: 20px 24px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    clip-path: polygon(0 0, calc(100% - 12px) 0, 100% 12px, 100% 100%, 12px 100%, 0 calc(100% - 12px));
}

div[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff8c, transparent);
    animation: scanbeam 3s linear infinite;
}

@keyframes scanbeam {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

div[data-testid="metric-container"]:hover {
    border-color: #00ff8c88;
    box-shadow: 0 0 30px rgba(0,255,140,0.15), inset 0 0 30px rgba(0,255,140,0.03);
    transform: translateY(-3px);
}

div[data-testid="metric-container"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #00ff8c99 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #00ff8c !important;
    text-shadow: 0 0 10px rgba(0,255,140,0.5);
}

/* Corner decorators on metric cards */
div[data-testid="metric-container"]::after {
    content: '◈';
    position: absolute;
    bottom: 8px; right: 10px;
    font-size: 10px;
    color: #00ff8c33;
    font-family: 'Share Tech Mono', monospace;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid #00ff8c22;
}

.stTabs [data-baseweb="tab"] {
    padding: 10px 24px;
    background-color: transparent;
    border: 1px solid #00ff8c22;
    border-bottom: none;
    border-radius: 2px 2px 0 0;
    color: #00ff8c66;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    transition: all 0.2s;
    clip-path: polygon(0 0, calc(100% - 8px) 0, 100% 8px, 100% 100%, 0 100%);
}

.stTabs [data-baseweb="tab"]:hover {
    color: #00ff8c;
    background: rgba(0,255,140,0.05);
    border-color: #00ff8c44;
}

.stTabs [aria-selected="true"] {
    background: rgba(0,255,140,0.1) !important;
    color: #00ff8c !important;
    border-color: #00ff8c !important;
    text-shadow: 0 0 10px rgba(0,255,140,0.6);
}

/* ─── Dataframe ─── */
[data-testid="stDataFrame"] {
    border: 1px solid #00ff8c22 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ─── Buttons ─── */
.stButton button {
    background: transparent !important;
    color: #00ff8c !important;
    border: 1px solid #00ff8c !important;
    border-radius: 2px !important;
    padding: 10px 24px !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 3px !important;
    transition: all 0.25s !important;
    clip-path: polygon(0 0, calc(100% - 10px) 0, 100% 10px, 100% 100%, 10px 100%, 0 calc(100% - 10px)) !important;
    position: relative;
}

.stButton button:hover {
    background: rgba(0,255,140,0.12) !important;
    box-shadow: 0 0 20px rgba(0,255,140,0.3), inset 0 0 20px rgba(0,255,140,0.05) !important;
    transform: translateY(-1px) !important;
    letter-spacing: 4px !important;
}

/* ─── Inputs / Selects ─── */
.stTextInput input,
.stTextArea textarea {
    background: #030d18 !important;
    border: 1px solid #00ff8c33 !important;
    border-radius: 2px !important;
    color: #00ff8c !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.9rem !important;
    caret-color: #00ff8c;
}

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #00ff8c !important;
    box-shadow: 0 0 0 1px #00ff8c44, 0 0 20px rgba(0,255,140,0.1) !important;
}

.stSelectbox [data-baseweb="select"] > div {
    background: #030d18 !important;
    border: 1px solid #00ff8c33 !important;
    border-radius: 2px !important;
    color: #00ff8c !important;
    font-family: 'Share Tech Mono', monospace !important;
}

/* ─── Info Box ─── */
.custom-info-box {
    background: #030d18;
    padding: 24px;
    border: 1px solid #00ff8c44;
    border-left: 3px solid #00ff8c;
    border-radius: 2px;
    margin-top: 20px;
    position: relative;
    clip-path: polygon(0 0, calc(100% - 14px) 0, 100% 14px, 100% 100%, 14px 100%, 0 calc(100% - 14px));
}

.custom-info-box::before {
    content: 'OUTPUT';
    position: absolute;
    top: -1px; right: 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 9px;
    color: #00ff8c;
    background: #030d18;
    padding: 0 6px;
    letter-spacing: 2px;
}

/* ─── Spinner / Status ─── */
.stSpinner > div > div {
    border-color: #00ff8c transparent transparent !important;
}

/* ─── Sidebar text ─── */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] span {
    color: #00ff8c88 !important;
    font-size: 0.8rem !important;
    line-height: 1.8 !important;
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 0.75rem !important;
    letter-spacing: 3px !important;
    color: #00ff8ccc !important;
    text-shadow: 0 0 8px rgba(0,255,140,0.3) !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff8c !important;
    background: transparent !important;
    border: 1px solid #00ff8c22 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

/* ─── Alert / Info ─── */
.stAlert {
    background: #030d18 !important;
    border: 1px solid #00ff8c33 !important;
    border-radius: 2px !important;
    font-family: 'Share Tech Mono', monospace !important;
    color: #00ff8c !important;
}

/* ─── Decorative HUD corners ─── */
.hud-box {
    position: relative;
    padding: 20px;
    border: 1px solid #00ff8c22;
    background: rgba(0,255,140,0.02);
    border-radius: 2px;
}
.hud-box::before, .hud-box::after {
    content: '';
    position: absolute;
    width: 12px; height: 12px;
    border-color: #00ff8c;
    border-style: solid;
}
.hud-box::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.hud-box::after  { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

/* ─── Section divider ─── */
hr {
    border: none !important;
    border-top: 1px solid #00ff8c22 !important;
    margin: 2rem 0 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #020810; }
::-webkit-scrollbar-thumb { background: #00ff8c44; border-radius: 0; }
::-webkit-scrollbar-thumb:hover { background: #00ff8c88; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 20px;'>
        <div style='font-family:"Share Tech Mono",monospace; font-size:2rem; color:#00ff8c; text-shadow:0 0 20px rgba(0,255,140,0.6);'>◈</div>
        <div style='font-family:"Share Tech Mono",monospace; font-size:0.65rem; letter-spacing:5px; color:#00ff8c99; text-transform:uppercase; margin-top:4px;'>Green AI Hub</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### SYS // DATASET")
    st.markdown("**SQuAD v1.1** — Stanford QA")
    st.markdown("### SYS // METRICS")
    st.markdown("- `EM` — Exact Match %")
    st.markdown("- `F1` — Token Overlap %")
    st.markdown("- `CO₂` — kg equivalent")
    st.markdown("- `Δt` — Duration (s)")
    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; color:#00ff8c44; line-height:1.8;'>
    > Ecological intelligence<br>
    > monitoring system v2.1<br>
    > STATUS: ONLINE ██████░░ 82%
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Main Title
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 2rem 0 1rem;'>
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:6px; color:#00ff8c66; margin-bottom:8px;'>◄ SYSTEM ACTIVE ►</div>
    <h1 style='font-family:"Exo 2",sans-serif; font-size:2.6rem; font-weight:900; letter-spacing:6px; color:#00ff8c; text-shadow:0 0 40px rgba(0,255,140,0.4), 0 0 80px rgba(0,255,140,0.15); margin:0;'>
        AI SUSTAINABILITY DASHBOARD
    </h1>
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.75rem; color:#00ff8c55; letter-spacing:3px; margin-top:10px;'>
        ECOLOGICAL IMPACT ANALYSIS // NLP QUESTION ANSWERING ARCHITECTURES
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Results
# ─────────────────────────────────────────────────────────────
csv_path = "results/experiment_results.csv"

if not os.path.exists(csv_path):
    st.error("🚨 experiment_results.csv not found! Please run the training and evaluation scripts first.")
    st.stop()

df = pd.read_csv(csv_path)

# ─── Auto-Heal CSV ───
df.columns = df.columns.str.strip()

if "Model" not in df.columns and len(df.columns) >= 5:
    df = pd.read_csv(csv_path, header=None)
    expected = ["Model", "Dataset Size", "Epochs", "Training Time (s)", "CO2 (kg)", "Validation Loss", "EM", "F1"]
    df.columns = expected[:len(df.columns)]

if "Epoch" in df.columns:
    df.rename(columns={"Epoch": "Epochs"}, inplace=True)
elif "epoch" in df.columns:
    df.rename(columns={"epoch": "Epochs"}, inplace=True)

if "Epochs" not in df.columns:
    df["Epochs"] = 1

if "EM_y" in df.columns:
    df["EM"] = df["EM_y"]
elif "EM_x" in df.columns:
    df["EM"] = df["EM_x"]

if "F1_y" in df.columns:
    df["F1"] = df["F1_y"]
elif "F1_x" in df.columns:
    df["F1"] = df["F1_x"]

df = df.loc[:, ~df.columns.str.contains('_x|_y')]
df["EM"] = df.get("EM", 0).fillna(0)
df["F1"] = df.get("F1", 0).fillna(0)

df["Carbon_Efficiency"] = df.apply(
    lambda row: row["F1"] / row["CO2 (kg)"] if row["CO2 (kg)"] > 0 else 0, axis=1
)

best_f1_val           = df["F1"].max() if len(df) > 0 else 0
best_accuracy_model   = df[df["F1"] == best_f1_val].iloc[0]["Model"] if len(df) > 0 else "N/A"
lowest_co2_val        = df["CO2 (kg)"].min() if len(df) > 0 else 0
lowest_carbon_model   = df[df["CO2 (kg)"] == lowest_co2_val].iloc[0]["Model"] if len(df) > 0 else "N/A"
best_eff_val          = df["Carbon_Efficiency"].max() if len(df) > 0 else 0
best_efficiency_model = df[df["Carbon_Efficiency"] == best_eff_val].iloc[0]["Model"] if len(df) > 0 else "N/A"

# ─────────────────────────────────────────────────────────────
# Plotly dark theme for all charts
# ─────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Share Tech Mono, monospace", color="rgba(0,255,140,0.6)", size=11),
    title_font=dict(family="Exo 2, sans-serif", color="#00ff8c", size=16),
    legend=dict(
        bgcolor="rgba(2,8,16,0.8)",
        bordercolor="rgba(0,255,140,0.13)",
        borderwidth=1
    ),
    xaxis=dict(
        gridcolor="rgba(0,255,140,0.07)",
        linecolor="rgba(0,255,140,0.13)",
        tickfont=dict(family="Share Tech Mono, monospace", color="rgba(0,255,140,0.4)", size=10)
    ),
    yaxis=dict(
        gridcolor="rgba(0,255,140,0.07)",
        linecolor="rgba(0,255,140,0.13)",
        tickfont=dict(family="Share Tech Mono, monospace", color="rgba(0,255,140,0.4)", size=10)
    ),
)

GREEN_SCALE = ["#003320", "#005533", "#007744", "#00aa66", "#00cc77", "#00ff8c", "#aaffcc"]

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["[ 01 ] BENCHMARKS", "[ 02 ] ECO-ANALYTICS", "[ 03 ] LIVE INFERENCE"])

# ════════════════════════════════
# TAB 1
# ════════════════════════════════
with tab1:
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:1.5rem;'>
    ▸ PERFORMANCE LEADER BOARD // TOP RANKED ARCHITECTURES
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("⬡ PEAK ACCURACY (F1)", best_accuracy_model, f"{best_f1_val:.4f}")
    col2.metric("⬡ MINIMAL FOOTPRINT", lowest_carbon_model, f"{lowest_co2_val:.5f} kg CO₂", delta_color="inverse")
    col3.metric("⬡ HIGHEST EFFICIENCY", best_efficiency_model, f"{best_eff_val:.2f} F1/kg")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:0.5rem;'>
    ▸ EMPIRICAL BENCHMARK DATASET // RAW TELEMETRY
    </div>
    """, unsafe_allow_html=True)

    columns_needed = ["Model", "Epochs", "Training Time (s)", "CO2 (kg)", "Validation Loss", "EM", "F1", "Carbon_Efficiency"]
    available_columns = [col for col in columns_needed if col in df.columns]
    display_df = df[available_columns].copy()
    for col in ["CO2 (kg)", "Validation Loss", "Carbon_Efficiency"]:
        if col in display_df:
            display_df[col] = display_df[col].apply(lambda x: f"{x:.6f}" if isinstance(x, (float, int)) else x)

    st.dataframe(display_df, use_container_width=True, height=250)

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("◈  RESEARCH INTERPRETATIONS // EXPAND TO DECODE"):
        st.markdown("""
        **◦ THE ORTHOGONALITY OF SUSTAINABILITY**
        A model maximizing F1/EM performance does not natively optimize computational or thermal efficiency.

        **◦ DISTILLATION IMPACT**
        *Student Distillation* creates a permanent architectural acceleration, driving up the payload-per-watt efficiency over prolonged batch executions.

        **◦ ADAPTER LIMITATIONS**
        *LoRA* immensely reduces VRAM allocation and training payload sizes, however evaluating LoRA boundaries requires aggressive adapter configurations to surpass natively distilled baseline heads.
        """)

# ════════════════════════════════
# TAB 2
# ════════════════════════════════
with tab2:
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:1.5rem;'>
    ▸ MULTIVARIATE TRADE-OFF VISUALIZATIONS // PARETO ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig1 = px.scatter(
            df, x="CO2 (kg)", y="F1", text="Model", size="F1",
            color="Carbon_Efficiency", color_continuous_scale=GREEN_SCALE,
            title="Pareto Frontier Configuration"
        )
        fig1.update_layout(**CHART_LAYOUT, xaxis_title="Carbon Emitted (kg CO₂)", yaxis_title="F1 Score")
        fig1.update_traces(
            textposition="top center",
            textfont=dict(color="#00ff8c", size=10, family="Share Tech Mono"),
            marker=dict(line=dict(width=1, color='#020810'))
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_chart2:
        fig2 = px.bar(
            df, x="Model", y="CO2 (kg)", color="CO2 (kg)",
            color_continuous_scale=GREEN_SCALE,
            title="Aggregate Training Emissions"
        )
        fig2.update_layout(**CHART_LAYOUT, xaxis_title="Architecture", yaxis_title="kg CO₂ eq.")
        fig2.update_traces(marker=dict(line=dict(width=1, color='#020810')))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:1.5rem;'>
    ▸ ALGORITHMIC YIELD & COMPUTE OVERHEAD // EFFICIENCY MATRIX
    </div>
    """, unsafe_allow_html=True)

    col_chart3, col_chart4 = st.columns(2)

    with col_chart3:
        fig3 = px.bar(
            df, x="Model", y="Carbon_Efficiency", color="Carbon_Efficiency",
            color_continuous_scale=GREEN_SCALE,
            title="Carbon Yield Ratio (F1 / kg CO₂)"
        )
        fig3.update_layout(**CHART_LAYOUT, xaxis_title="Architecture", yaxis_title="Efficiency (F1/kg)")
        fig3.update_traces(marker=dict(line=dict(width=1, color='#020810')))
        st.plotly_chart(fig3, use_container_width=True)

    with col_chart4:
        fig4 = px.scatter(
            df, x="Training Time (s)", y="CO2 (kg)", color="Model",
            size="F1", size_max=45,
            title="Compute Wallclock vs Ecological Overhead"
        )
        fig4.update_layout(**CHART_LAYOUT, xaxis_title="Training Duration (s)", yaxis_title="Carbon Burden (kg CO₂)")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:1.5rem;'>
    ▸ TEMPORAL EPOCH TRAJECTORY // LEARNING CURVE ANALYSIS
    </div>
    """, unsafe_allow_html=True)

    epoch_df = df.sort_values(by=["Model", "Epochs"]).copy()
    col_chart5, col_chart6 = st.columns(2)

    with col_chart5:
        fig5 = px.line(
            epoch_df, x="Epochs", y="F1", color="Model", markers=True,
            title="Epoch-by-Epoch Accuracy Trajectory"
        )
        fig5.update_layout(**CHART_LAYOUT, xaxis_title="Training Epoch", yaxis_title="Validation F1")
        fig5.update_xaxes(type='category')
        st.plotly_chart(fig5, use_container_width=True)

    with col_chart6:
        fig6 = px.line(
            epoch_df, x="Epochs", y="CO2 (kg)", color="Model", markers=True,
            title="Carbon Burden Accumulation per Epoch"
        )
        fig6.update_layout(**CHART_LAYOUT, xaxis_title="Training Epoch", yaxis_title="CO₂ Payload (kg)")
        fig6.update_xaxes(type='category')
        st.plotly_chart(fig6, use_container_width=True)

# ════════════════════════════════
# TAB 3  —  FIXED INFERENCE
# ════════════════════════════════
with tab3:
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px; color:#00ff8c55; margin-bottom:1.5rem;'>
    ▸ LIVE SUSTAINABLE NLP CLOUD ENGINE // NEURAL INFERENCE TERMINAL
    </div>
    """, unsafe_allow_html=True)

    # ── FIX 1: separate cache functions for QA vs Seq2Seq models ──────────────
    @st.cache_resource
    def load_qa_model(model_path):
        """Load a standard extractive QA model (BERT / DistilBERT / LoRA)."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_path)
        model.eval()
        return tokenizer, model

    @st.cache_resource
    def load_seq2seq_model(model_path):
        """Load a generative Seq2Seq model (T5)."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        return tokenizer, model

    model_options = {
        "DistilBERT Pioneer":        "models/distilbert",
        "BERT Base Engine":          "models/bert",
        "LoRA Infused BERT":         "models/lora_bert",
        "Distilled Teacher-Student": "models/distilled_student",
        "T5 Sequential Generator":   "models/t5",
    }

    # Which names are T5-style generative models
    SEQ2SEQ_MODELS = {"T5 Sequential Generator"}

    col_input, col_spacer, col_output = st.columns([1.2, 0.1, 1])

    with col_input:
        st.markdown("""
        <div style='font-family:"Share Tech Mono",monospace; font-size:0.7rem; letter-spacing:3px;
            color:#00ff8c55; margin-bottom:12px;'>▸ INPUT TERMINAL</div>
        """, unsafe_allow_html=True)
        selected_model_name = st.selectbox("[ ARCHITECTURE ]", list(model_options.keys()))
        context  = st.text_area("[ PASSAGE CONTEXT ]", height=150,
            placeholder="The Apollo 11 mission landed on the Moon in 1969...")
        question = st.text_input("[ QUERY ]",
            placeholder="When did Apollo 11 execute its landing?")
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("▶  INITIALIZE NEURAL INFERENCE", use_container_width=True)

    with col_output:
        st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)

        if run_btn:
            if not question.strip() or not context.strip():
                st.warning("⚠ CONSTRAINT: Furnish both a valid passage and query.")
            else:
                model_path = model_options[selected_model_name]
                is_seq2seq = selected_model_name in SEQ2SEQ_MODELS

                if not os.path.exists(model_path):
                    st.error(f"⛔ NODE UNREACHABLE: {model_path} checkpoint not found.")
                else:
                    with st.spinner(f"ROUTING THROUGH {selected_model_name.upper()}..."):
                        try:
                            # ── FIX 2: load the correct model class ──────────
                            if is_seq2seq:
                                tokenizer, model = load_seq2seq_model(model_path)
                            else:
                                tokenizer, model = load_qa_model(model_path)

                            # ── FIX 3: tokenise once, outside the tracker ─────
                            if is_seq2seq:
                                input_text = f"question: {question}  context: {context}"
                                inputs = tokenizer(
                                    input_text,
                                    return_tensors="pt",
                                    truncation=True,
                                    max_length=512,
                                )
                            else:
                                inputs = tokenizer(
                                    question,
                                    context,
                                    return_tensors="pt",
                                    truncation="only_second",
                                    max_length=384,
                                    return_offsets_mapping=False,
                                )

                            # ── FIX 4: single forward pass inside tracker ─────
                            tracker = EmissionsTracker(
                                measure_power_secs=0.1,
                                log_level="error",
                                save_to_file=False,
                            )
                            tracker.start()
                            start_time = time.time()

                            with torch.no_grad():
                                if is_seq2seq:
                                    output_ids = model.generate(
                                        **inputs,
                                        max_new_tokens=64,
                                    )
                                else:
                                    outputs = model(**inputs)

                            end_time  = time.time()
                            emissions = tracker.stop()

                            # ── FIX 5: safe emission value ────────────────────
                            if emissions is None or (isinstance(emissions, float) and math.isnan(emissions)):
                                emissions = 0.0

                            inference_time = end_time - start_time

                            # ── FIX 6: correct answer extraction ─────────────
                            if is_seq2seq:
                                answer = tokenizer.decode(
                                    output_ids[0],
                                    skip_special_tokens=True,
                                ).strip()
                            else:
                                input_ids   = inputs["input_ids"][0]
                                seq_len     = input_ids.size(0)

                                start_logits = outputs.start_logits[0]
                                end_logits   = outputs.end_logits[0]

                                # Mask CLS and padding; find best valid span
                                start_logits[0] = -float("inf")   # suppress [CLS]
                                end_logits[0]   = -float("inf")

                                # ── FIX 7: pick best (start, end) pair where
                                #           start <= end and end < seq_len ──────
                                best_score  = -float("inf")
                                best_start  = 0
                                best_end    = 0
                                max_answer_length = 30

                                for s in range(1, seq_len):
                                    for e in range(s, min(s + max_answer_length, seq_len)):
                                        score = start_logits[s].item() + end_logits[e].item()
                                        if score > best_score:
                                            best_score = score
                                            best_start = s
                                            best_end   = e

                                answer = tokenizer.decode(
                                    input_ids[best_start : best_end + 1],
                                    skip_special_tokens=True,
                                ).strip()

                            st.success("✔  NEURAL PIPELINE COMPLETE")
                            st.markdown(f"""
                            <div class="custom-info-box">
                                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                                    letter-spacing:3px; color:#00ff8c66; margin-bottom:10px;">
                                    RETRIEVED SYNTAX PAYLOAD
                                </div>
                                <div style="font-family:'Exo 2',sans-serif; font-size:1.4rem;
                                    font-weight:700; color:#00ff8c;
                                    text-shadow: 0 0 20px rgba(0,255,140,0.4);
                                    margin-bottom:16px;">
                                    {answer if answer else "NO VALID CONTEXT MATCH"}
                                </div>
                                <div style="border-top:1px solid #00ff8c22; padding-top:12px;
                                    display:flex; justify-content:space-between; gap:12px;">
                                    <span style="font-family:'Share Tech Mono',monospace;
                                        font-size:0.75rem; color:#00ff8c66;">
                                        Δt <span style="color:#00ff8c;">{inference_time*1000:.1f} ms</span>
                                    </span>
                                    <span style="font-family:'Share Tech Mono',monospace;
                                        font-size:0.75rem; color:#00ff8c66;">
                                        CO₂ <span style="color:#00ff8c;">{emissions*1000:.6f} mg</span>
                                    </span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"INFERENCE EXCEPTION: {e}")
        else:
            st.markdown("""
            <div style='font-family:"Share Tech Mono",monospace; font-size:0.8rem;
                color:#00ff8c33; padding:40px 20px; text-align:center;
                border:1px solid #00ff8c11; border-radius:2px; margin-top:40px;
                line-height:2;'>
                AWAITING INPUT STREAM<br>
                <span style='font-size:0.6rem; letter-spacing:4px; color:#00ff8c22;'>
                ░░░░░░░░░░ IDLE ░░░░░░░░░░
                </span>
            </div>
            """, unsafe_allow_html=True)


