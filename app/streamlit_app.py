"""
streamlit_app.py
----------------
Dashboard MMM Multi-Market — Design professionnel
Usage: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from src.data.data_loader import load_market_data, load_all_markets, split_train_test
from src.models.bayesian_mmm import BayesianMMM
from src.evaluation.metrics import compute_all_metrics

# ── CONFIG PAGE ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MMM Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DESIGN SYSTEM ─────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#0D0F14",
    "surface":     "#13161E",
    "surface2":    "#1A1E2A",
    "border":      "#252A38",
    "accent":      "#4F7EFF",
    "accent2":     "#00D4AA",
    "accent3":     "#FF6B6B",
    "text":        "#E8EAF0",
    "text_muted":  "#6B7280",
    "channels": {
        "tv":       "#4F7EFF",
        "facebook": "#00D4AA",
        "search":   "#FFB547",
        "ooh":      "#FF6B6B",
        "print":    "#A78BFA",
    }
}

CHANNEL_LABELS = {
    "tv":       "TV",
    "facebook": "Facebook",
    "search":   "Search",
    "ooh":      "OOH",
    "print":    "Print",
}

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {{
      font-family: 'DM Sans', sans-serif;
      background-color: {COLORS['bg']};
      color: {COLORS['text']};
  }}
  .stApp {{ background-color: {COLORS['bg']}; }}

  /* Sidebar */
  [data-testid="stSidebar"] {{
      background-color: {COLORS['surface']};
      border-right: 1px solid {COLORS['border']};
  }}
  [data-testid="stSidebar"] * {{ color: {COLORS['text']} !important; }}

  /* Inputs */
  .stSelectbox > div > div,
  .stMultiSelect > div > div {{
      background-color: {COLORS['surface2']};
      border: 1px solid {COLORS['border']};
      color: {COLORS['text']};
      border-radius: 6px;
  }}
  .stSlider > div {{ color: {COLORS['text']}; }}

  /* Remove default padding */
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }}

  /* Metric cards */
  .metric-card {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 10px;
      padding: 20px 24px;
      height: 100%;
  }}
  .metric-label {{
      font-size: 11px;
      font-weight: 500;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: {COLORS['text_muted']};
      margin-bottom: 8px;
  }}
  .metric-value {{
      font-size: 28px;
      font-weight: 600;
      color: {COLORS['text']};
      font-family: 'DM Mono', monospace;
      line-height: 1;
  }}
  .metric-delta {{
      font-size: 12px;
      margin-top: 6px;
      font-family: 'DM Mono', monospace;
  }}
  .delta-pos {{ color: {COLORS['accent2']}; }}
  .delta-neg {{ color: {COLORS['accent3']}; }}
  .delta-neu {{ color: {COLORS['text_muted']}; }}

  /* Section titles */
  .section-title {{
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: {COLORS['text_muted']};
      border-bottom: 1px solid {COLORS['border']};
      padding-bottom: 10px;
      margin-bottom: 20px;
  }}

  /* Page title */
  .page-header {{
      display: flex;
      align-items: baseline;
      gap: 12px;
      margin-bottom: 28px;
  }}
  .page-title {{
      font-size: 22px;
      font-weight: 600;
      color: {COLORS['text']};
  }}
  .page-subtitle {{
      font-size: 13px;
      color: {COLORS['text_muted']};
  }}

  /* Nav pills */
  .nav-pill {{
      display: inline-block;
      padding: 5px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 500;
      background: {COLORS['surface2']};
      border: 1px solid {COLORS['border']};
      color: {COLORS['text_muted']};
      margin-right: 6px;
  }}
  .nav-pill-active {{
      background: {COLORS['accent']};
      border-color: {COLORS['accent']};
      color: white;
  }}

  /* Plotly container */
  .plot-container {{
      background: {COLORS['surface']};
      border: 1px solid {COLORS['border']};
      border-radius: 10px;
      padding: 4px;
  }}

  /* Scrollbar */
  ::-webkit-scrollbar {{ width: 6px; height: 6px; }}
  ::-webkit-scrollbar-track {{ background: {COLORS['bg']}; }}
  ::-webkit-scrollbar-thumb {{ background: {COLORS['border']}; border-radius: 3px; }}

  /* Hide Streamlit branding */
  #MainMenu, footer, header {{ visibility: hidden; }}

  /* Table */
  .stDataFrame {{ border: 1px solid {COLORS['border']}; border-radius: 8px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)


# ── PLOTLY THEME ──────────────────────────────────────────────────────────────

def hex_to_rgba(hex_color: str, alpha: float = 1.0) -> str:
    """Convertit un hex color en rgba() compatible Plotly."""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color=COLORS["text"], size=12),
    xaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
               zerolinecolor=COLORS["border"]),
    yaxis=dict(gridcolor=COLORS["border"], linecolor=COLORS["border"],
               zerolinecolor=COLORS["border"]),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=COLORS["border"],
                borderwidth=1),
    margin=dict(l=16, r=16, t=36, b=16),
    hoverlabel=dict(bgcolor=COLORS["surface2"], bordercolor=COLORS["border"],
                    font_color=COLORS["text"]),
)


# ── CACHE & DATA ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def get_all_data():
    return load_all_markets()

@st.cache_resource
def get_model(market: str):
    data = get_all_data()
    df   = data[market] if isinstance(data, dict) else data[data["market"] == market]
    df_train, df_test = split_train_test(df, test_ratio=0.2)
    model = BayesianMMM(market=market)
    model.fit(df_train, draws=300, tune=300, chains=2, random_seed=42)
    return model, df_train, df_test


# ── HELPERS ───────────────────────────────────────────────────────────────────
def card_metric(label, value, delta=None, delta_type="neu"):
    delta_html = ""
    if delta is not None:
        delta_html = f'<div class="metric-delta delta-{delta_type}">{delta}</div>'
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      {delta_html}
    </div>"""

def section(title):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)

def fmt_eur(v):
    if v >= 1e6:   return f"€{v/1e6:.2f}M"
    if v >= 1e3:   return f"€{v/1e3:.0f}K"
    return f"€{v:.0f}"

def apply_layout(fig, title="", height=340):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font_size=13,
                      font_color=COLORS["text_muted"], x=0, xanchor="left"),
                      height=height)
    return fig


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
ALL_MARKETS = ["FR", "DE", "UK", "IT", "ES", "NL", "BE", "PL", "SE", "NO"]
MARKET_NAMES = {
    "FR":"France","DE":"Germany","UK":"United Kingdom","IT":"Italy",
    "ES":"Spain","NL":"Netherlands","BE":"Belgium","PL":"Poland",
    "SE":"Sweden","NO":"Norway"
}

with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
      <div style="font-size:18px;font-weight:700;color:#E8EAF0;letter-spacing:-0.3px;">MMM Analytics</div>
      <div style="font-size:11px;color:#6B7280;margin-top:3px;">Multi-Market Bayesian Model</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigation",
        ["Overview", "Model Performance", "ROI Analysis", "Budget Optimizer", "Market Comparison"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#252A38;margin:16px 0'>", unsafe_allow_html=True)

    market = st.selectbox(
        "Market",
        ALL_MARKETS,
        format_func=lambda x: f"{x} — {MARKET_NAMES[x]}",
    )

    st.markdown("<hr style='border-color:#252A38;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:11px;color:#6B7280;line-height:1.7;">
      <div style="color:#E8EAF0;font-weight:500;margin-bottom:6px;">Dataset</div>
      10 markets · 208 weeks<br>
      5 channels · Bayesian NUTS<br>
      Seed 42 · OLS fallback
    </div>
    """, unsafe_allow_html=True)


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
data_all = get_all_data()
if isinstance(data_all, dict):
    df_market = data_all[market]
else:
    df_market = data_all[data_all["market"] == market].copy()

df_train, df_test = split_train_test(df_market, test_ratio=0.2)
SPEND_COLS = ["tv_spend","facebook_spend","search_spend","ooh_spend","print_spend"]
CHANNELS   = ["tv","facebook","search","ooh","print"]


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">{MARKET_NAMES[market]}</div>
      <div class="page-subtitle">Market Overview · 208 weeks</div>
    </div>""", unsafe_allow_html=True)

    # KPIs
    total_rev   = df_market["revenue"].sum()
    avg_rev     = df_market["revenue"].mean()
    total_spend = df_market[SPEND_COLS].sum().sum()
    global_roi  = total_rev / total_spend if total_spend > 0 else 0
    top_channel = df_market[SPEND_COLS].sum().idxmax().replace("_spend","")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card_metric("Total Revenue", fmt_eur(total_rev), f"avg {fmt_eur(avg_rev)}/wk", "pos"), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Total Spend",   fmt_eur(total_spend), f"avg {fmt_eur(total_spend/208)}/wk", "neu"), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("Global ROI",    f"{global_roi:.2f}x", "revenue / spend", "pos"), unsafe_allow_html=True)
    with c4: st.markdown(card_metric("Top Channel",   CHANNEL_LABELS[top_channel], "by total spend", "neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Revenue time series
    col_l, col_r = st.columns([2, 1])

    with col_l:
        section("Revenue Over Time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_market["date"], y=df_market["revenue"],
            fill="tozeroy",
            fillcolor="rgba(79,126,255,0.08)",
            line=dict(color=COLORS["accent"], width=2),
            name="Revenue",
            hovertemplate="<b>%{x|%b %Y}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=280)
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Channel Spend Mix")
        spend_by_ch = {CHANNEL_LABELS[c]: df_market[f"{c}_spend"].sum() for c in CHANNELS}
        fig = go.Figure(go.Pie(
            labels=list(spend_by_ch.keys()),
            values=list(spend_by_ch.values()),
            hole=0.62,
            marker_colors=[COLORS["channels"][c] for c in CHANNELS],
            textinfo="percent",
            textfont_size=11,
            hovertemplate="<b>%{label}</b><br>%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        apply_layout(fig, height=280)
        fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.05))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Weekly spend by channel
    section("Weekly Spend by Channel")
    fig = go.Figure()
    for ch in CHANNELS:
        fig.add_trace(go.Scatter(
            x=df_market["date"], y=df_market[f"{ch}_spend"],
            stackgroup="one",
            name=CHANNEL_LABELS[ch],
            line=dict(width=0),
            fillcolor=hex_to_rgba(COLORS["channels"][ch], 0.8),
            hovertemplate=f"<b>{CHANNEL_LABELS[ch]}</b><br>€%{{y:,.0f}}<extra></extra>",
        ))
    apply_layout(fig, height=240)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">Model Performance</div>
      <div class="page-subtitle">{MARKET_NAMES[market]} · Bayesian MMM</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Training model…"):
        model, df_tr, df_te = get_model(market)

    y_pred_train = model.predict(df_tr)
    y_pred_test  = model.predict(df_te)
    m_train = compute_all_metrics(df_tr["revenue"].values, y_pred_train)
    m_test  = compute_all_metrics(df_te["revenue"].values, y_pred_test)

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    r2_ok   = m_test["r2"]   >= 0.85
    mape_ok = m_test["mape"] <= 15
    with c1: st.markdown(card_metric("R² (test)",  f"{m_test['r2']:.3f}",  "target ≥ 0.85", "pos" if r2_ok else "neg"),   unsafe_allow_html=True)
    with c2: st.markdown(card_metric("MAPE (test)", f"{m_test['mape']:.1f}%", "target ≤ 15%",  "pos" if mape_ok else "neg"), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("RMSE (test)", fmt_eur(m_test["rmse"]), "root mean sq error", "neu"), unsafe_allow_html=True)
    with c4: st.markdown(card_metric("R² (train)", f"{m_train['r2']:.3f}", "in-sample fit", "neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted — Test
    col_l, col_r = st.columns([3, 1])
    with col_l:
        section("Actual vs Predicted — Test Set")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_te["date"], y=df_te["revenue"],
            name="Actual", line=dict(color=COLORS["text"], width=1.5),
            hovertemplate="<b>Actual</b><br>€%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_te["date"], y=y_pred_test,
            name="Predicted", line=dict(color=COLORS["accent"], width=2, dash="dot"),
            hovertemplate="<b>Predicted</b><br>€%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Fit Quality")
        corr = np.corrcoef(df_te["revenue"].values, y_pred_test)[0, 1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_te["revenue"].values, y=y_pred_test,
            mode="markers",
            marker=dict(color=COLORS["accent"], size=5, opacity=0.7),
            name="Obs vs Pred",
            hovertemplate="<b>Actual</b>: €%{x:,.0f}<br><b>Pred</b>: €%{y:,.0f}<extra></extra>",
        ))
        lim = [min(df_te["revenue"].min(), y_pred_test.min()) * 0.97,
               max(df_te["revenue"].max(), y_pred_test.max()) * 1.03]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
                      line=dict(color=COLORS["accent3"], width=1, dash="dash"),
                      name="Perfect fit", showlegend=False))
        apply_layout(fig, height=300)
        fig.update_xaxes(title="Actual (€)")
        fig.update_yaxes(title="Predicted (€)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Residuals
    section("Residuals Distribution")
    residuals = df_te["revenue"].values - y_pred_test
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals over time", "Distribution"])
    fig.add_trace(go.Scatter(
        x=df_te["date"], y=residuals, mode="lines+markers",
        line=dict(color=COLORS["accent2"], width=1.5),
        marker=dict(size=3), name="Residual",
        hovertemplate="€%{y:,.0f}<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=0, line_color=COLORS["accent3"], line_dash="dash", row=1, col=1)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=20,
        marker_color=COLORS["accent"], opacity=0.7, name="Distribution",
    ), row=1, col=2)
    fig.update_layout(**PLOTLY_LAYOUT, height=260, showlegend=False)
    fig.update_xaxes(gridcolor=COLORS["border"])
    fig.update_yaxes(gridcolor=COLORS["border"])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ROI ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
elif page == "ROI Analysis":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">ROI Analysis</div>
      <div class="page-subtitle">{MARKET_NAMES[market]} · Channel Efficiency</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Computing ROI…"):
        model, df_tr, df_te = get_model(market)

    roi_df       = model.get_roi(df_tr)
    contributions = model.get_contributions(df_tr)

    # ROI KPIs
    cols = st.columns(5)
    for i, ch in enumerate(CHANNELS):
        row = roi_df[roi_df["channel"] == ch]
        roi_val = row["roi"].values[0] if not row.empty else 0
        with cols[i]:
            color = "pos" if roi_val >= 2 else "neu" if roi_val >= 1 else "neg"
            st.markdown(card_metric(CHANNEL_LABELS[ch], f"{roi_val:.2f}x",
                "ROI ratio", color), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])

    with col_l:
        section("ROI by Channel")
        roi_sorted = roi_df.sort_values("roi", ascending=True)
        fig = go.Figure(go.Bar(
            x=roi_sorted["roi"],
            y=[CHANNEL_LABELS[c] for c in roi_sorted["channel"]],
            orientation="h",
            marker=dict(
                color=[COLORS["channels"][c] for c in roi_sorted["channel"]],
                line=dict(width=0),
            ),
            text=[f"{v:.2f}x" for v in roi_sorted["roi"]],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=11, family="DM Mono"),
            hovertemplate="<b>%{y}</b><br>ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=1.0, line_color=COLORS["accent3"], line_dash="dash",
                      annotation_text="break-even", annotation_font_color=COLORS["text_muted"])
        apply_layout(fig, height=300)
        fig.update_xaxes(title="ROI (revenue / spend)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("Revenue Contribution Breakdown")
        contrib_mean = {}
        for ch in CHANNELS:
            if ch in contributions.columns:
                contrib_mean[CHANNEL_LABELS[ch]] = contributions[ch].mean()
        if "base" in contributions.columns:
            contrib_mean["Base"] = contributions["base"].mean()

        labels = list(contrib_mean.keys())
        values = list(contrib_mean.values())
        ch_colors = [COLORS["channels"].get(ch.lower(), COLORS["text_muted"])
                     for ch in ["tv","facebook","search","ooh","print"]] + [COLORS["border"]]

        fig = go.Figure(go.Pie(
            labels=labels, values=values,
            hole=0.55,
            marker_colors=ch_colors[:len(labels)],
            textinfo="percent+label",
            textfont_size=11,
            hovertemplate="<b>%{label}</b><br>€%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        apply_layout(fig, height=300)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Contributions over time
    section("Channel Contributions Over Time")
    fig = go.Figure()
    for ch in CHANNELS:
        if ch in contributions.columns:
            fig.add_trace(go.Scatter(
                x=contributions["date"] if "date" in contributions.columns else df_tr["date"],
                y=contributions[ch],
                stackgroup="one",
                name=CHANNEL_LABELS[ch],
                line=dict(width=0),
                fillcolor=hex_to_rgba(COLORS["channels"][ch], 0.73),
                hovertemplate=f"<b>{CHANNEL_LABELS[ch]}</b><br>€%{{y:,.0f}}<extra></extra>",
            ))
    apply_layout(fig, height=260)
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — BUDGET OPTIMIZER
# ════════════════════════════════════════════════════════════════════════════
elif page == "Budget Optimizer":
    st.markdown(f"""
    <div class="page-header">
      <div class="page-title">Budget Optimizer</div>
      <div class="page-subtitle">{MARKET_NAMES[market]} · Allocation Simulation</div>
    </div>""", unsafe_allow_html=True)

    with st.spinner("Loading model…"):
        model, df_tr, df_te = get_model(market)

    roi_df = model.get_roi(df_tr)
    roi_map = {row["channel"]: row["roi"] for _, row in roi_df.iterrows()}

    current_spend = {ch: df_tr[f"{ch}_spend"].mean() for ch in CHANNELS}
    current_total = sum(current_spend.values())

    col_ctrl, col_result = st.columns([1, 2])

    with col_ctrl:
        section("Budget Parameters")
        total_budget = st.slider(
            "Weekly Budget (€)",
            min_value=int(current_total * 0.5),
            max_value=int(current_total * 2.0),
            value=int(current_total),
            step=1000,
            format="€%d",
        )

        st.markdown("<div style='margin-top:16px;font-size:11px;color:#6B7280;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px;'>Channel Constraints</div>", unsafe_allow_html=True)
        constraints = {}
        for ch in CHANNELS:
            constraints[ch] = st.slider(
                f"{CHANNEL_LABELS[ch]} min %",
                0, 40,
                value={"tv": 20, "facebook": 15, "search": 15, "ooh": 10, "print": 5}[ch],
                step=5,
                key=f"min_{ch}",
            )

    with col_result:
        # Compute optimal allocation (ROI-weighted)
        total_roi = sum(roi_map.get(ch, 1) * (1 - constraints[ch]/100) for ch in CHANNELS)
        min_spend = {ch: total_budget * constraints[ch] / 100 for ch in CHANNELS}
        remaining = total_budget - sum(min_spend.values())

        if remaining < 0:
            st.warning("Constraints exceed total budget — reduce minimums.")
            allocation = {ch: total_budget / 5 for ch in CHANNELS}
        else:
            roi_sum = sum(roi_map.get(ch, 1) for ch in CHANNELS)
            allocation = {
                ch: min_spend[ch] + remaining * (roi_map.get(ch, 1) / roi_sum)
                for ch in CHANNELS
            }

        current_revenue = sum(current_spend[ch] * roi_map.get(ch, 1) for ch in CHANNELS)
        optimal_revenue = sum(allocation[ch] * roi_map.get(ch, 1) for ch in CHANNELS)
        revenue_delta   = (optimal_revenue - current_revenue) / current_revenue * 100

        section("Optimal Allocation")
        c1, c2, c3 = st.columns(3)
        with c1: st.markdown(card_metric("Total Budget", fmt_eur(total_budget), "weekly", "neu"), unsafe_allow_html=True)
        with c2: st.markdown(card_metric("Est. Revenue", fmt_eur(optimal_revenue), f"{revenue_delta:+.1f}% vs current", "pos" if revenue_delta > 0 else "neg"), unsafe_allow_html=True)
        with c3: st.markdown(card_metric("Blended ROI", f"{optimal_revenue/total_budget:.2f}x", "revenue / spend", "neu"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Comparison chart
        categories = [CHANNEL_LABELS[ch] for ch in CHANNELS]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current", x=categories,
            y=[current_spend[ch] for ch in CHANNELS],
            marker_color=COLORS["surface2"],
            marker_line_color=COLORS["border"], marker_line_width=1,
            hovertemplate="<b>Current %{x}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        fig.add_trace(go.Bar(
            name="Optimal", x=categories,
            y=[allocation[ch] for ch in CHANNELS],
            marker_color=[COLORS["channels"][ch] for ch in CHANNELS],
            hovertemplate="<b>Optimal %{x}</b><br>€%{y:,.0f}<extra></extra>",
        ))
        apply_layout(fig, height=280)
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_yaxes(title="Weekly Spend (€)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Breakdown table
        alloc_df = pd.DataFrame([{
            "Channel":    CHANNEL_LABELS[ch],
            "Current (€)":  f"€{current_spend[ch]:,.0f}",
            "Optimal (€)":  f"€{allocation[ch]:,.0f}",
            "Delta":        f"{(allocation[ch]-current_spend[ch])/current_spend[ch]*100:+.1f}%",
            "ROI":          f"{roi_map.get(ch,0):.2f}x",
        } for ch in CHANNELS])
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MARKET COMPARISON
# ════════════════════════════════════════════════════════════════════════════
elif page == "Market Comparison":
    st.markdown("""
    <div class="page-header">
      <div class="page-title">Market Comparison</div>
      <div class="page-subtitle">All Markets · Cross-Market Analysis</div>
    </div>""", unsafe_allow_html=True)

    data_all_dict = get_all_data()

    # Aggregate stats
    rows = []
    for mkt in ALL_MARKETS:
        df_m = data_all_dict[mkt] if isinstance(data_all_dict, dict) else data_all_dict[data_all_dict["market"] == mkt]
        total_rev   = df_m["revenue"].sum()
        total_spend = df_m[SPEND_COLS].sum().sum()
        rows.append({
            "market":      mkt,
            "name":        MARKET_NAMES[mkt],
            "revenue":     total_rev,
            "spend":       total_spend,
            "roi":         total_rev / total_spend if total_spend > 0 else 0,
            "avg_revenue": df_m["revenue"].mean(),
        })
    summary = pd.DataFrame(rows).sort_values("revenue", ascending=False)

    # Top KPIs
    best_roi = summary.loc[summary["roi"].idxmax()]
    best_rev = summary.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(card_metric("Markets Tracked", "10", "EU panel", "neu"), unsafe_allow_html=True)
    with c2: st.markdown(card_metric("Total Revenue",   fmt_eur(summary["revenue"].sum()), "all markets", "pos"), unsafe_allow_html=True)
    with c3: st.markdown(card_metric("Best ROI Market", best_roi["market"],  f"{best_roi['roi']:.2f}x", "pos"), unsafe_allow_html=True)
    with c4: st.markdown(card_metric("Top Revenue",     best_rev["market"],  fmt_eur(best_rev["revenue"]), "pos"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])

    with col_l:
        section("Revenue vs Spend by Market")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=summary["spend"], y=summary["revenue"],
            mode="markers+text",
            text=summary["market"],
            textposition="top center",
            textfont=dict(size=11, color=COLORS["text"]),
            marker=dict(
                size=summary["avg_revenue"] / summary["avg_revenue"].max() * 30 + 8,
                color=[COLORS["accent"] if r >= summary["roi"].mean() else COLORS["text_muted"]
                       for r in summary["roi"]],
                line=dict(color=COLORS["border"], width=1),
            ),
            hovertemplate="<b>%{text}</b><br>Revenue: €%{y:,.0f}<br>Spend: €%{x:,.0f}<extra></extra>",
            showlegend=False,
        ))
        apply_layout(fig, height=340)
        fig.update_xaxes(title="Total Spend (€)")
        fig.update_yaxes(title="Total Revenue (€)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        section("ROI Ranking")
        roi_sorted = summary.sort_values("roi", ascending=True)
        fig = go.Figure(go.Bar(
            x=roi_sorted["roi"],
            y=roi_sorted["market"],
            orientation="h",
            marker=dict(
                color=[COLORS["accent"] if r >= summary["roi"].mean()
                       else COLORS["text_muted"] for r in roi_sorted["roi"]],
                line=dict(width=0),
            ),
            text=[f"{v:.2f}x" for v in roi_sorted["roi"]],
            textposition="outside",
            textfont=dict(color=COLORS["text"], size=10, family="DM Mono"),
            hovertemplate="<b>%{y}</b><br>ROI: %{x:.3f}x<extra></extra>",
        ))
        fig.add_vline(x=summary["roi"].mean(), line_color=COLORS["accent2"],
                      line_dash="dash",
                      annotation_text=f"avg {summary['roi'].mean():.2f}x",
                      annotation_font_color=COLORS["accent2"], annotation_font_size=10)
        apply_layout(fig, height=340)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Spend heatmap
    section("Channel Spend Distribution by Market")
    heatmap_data = []
    for mkt in ALL_MARKETS:
        df_m = data_all_dict[mkt] if isinstance(data_all_dict, dict) else data_all_dict[data_all_dict["market"] == mkt]
        row = {}
        for ch in CHANNELS:
            row[CHANNEL_LABELS[ch]] = df_m[f"{ch}_spend"].mean()
        heatmap_data.append(row)

    hmap_df = pd.DataFrame(heatmap_data, index=ALL_MARKETS)
    hmap_norm = hmap_df.div(hmap_df.max(axis=0), axis=1)

    fig = go.Figure(go.Heatmap(
        z=hmap_norm.values,
        x=hmap_norm.columns.tolist(),
        y=ALL_MARKETS,
        colorscale=[[0, COLORS["surface"]], [0.5, hex_to_rgba(COLORS["accent"], 0.53)], [1, COLORS["accent"]]],
        text=[[fmt_eur(hmap_df.loc[mkt, ch]) for ch in hmap_df.columns] for mkt in ALL_MARKETS],
        texttemplate="%{text}",
        textfont_size=10,
        showscale=True,
        hovertemplate="<b>%{y} — %{x}</b><br>%{text}<extra></extra>",
    ))
    apply_layout(fig, height=320)
    fig.update_layout(margin=dict(l=40, r=60, t=30, b=16))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})