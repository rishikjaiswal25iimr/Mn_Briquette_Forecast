"""
Mn Briquette Hybrid Forecasting — Streamlit Dashboard  ·  v4.0
=============================================================
LIGHTWEIGHT web app — loads pre-computed CSVs only.
"""

from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mn Briquette Price Forecasting · v4.0",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light Theme Colors
C_ACTUAL   = "#FF9800" # Orange for Prices
C_HYBRID   = "#2196F3" # Blue for Index
C_FUTURE   = "#4CAF50" # Green for Future
C_MARKET   = "#E91E63" # Bright Magenta for Market Data
C_REG1     = "rgba(244, 67, 54, 0.15)" # Light Red for Squeeze
C_GRID     = "#E0E0E0" # Light Grey Grid
C_TEXT     = "#333333" # Dark Grey Text

OUTPUT_DIR = "./outputs"

HORIZON_OPTIONS = {
    "4 weeks (1 month)":     4,
    "12 weeks (3 months)":  12,
    "26 weeks (6 months)":  26,
    "52 weeks (1 year)":    52,
    "104 weeks (2 years)": 104,
    "156 weeks (3 years)": 156,
}

SHOCK_ANNOTATIONS = {
    "2018-07-06": "US 301 Tariff",
    "2021-03-01": "Cartel Cut",
    "2021-09-15": "Env Audit",
    "2024-03-01": "Ore Deficit Begins",
    "2025-04-01": "Cartel '25",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADERS (No Cache - forces reading latest files)
# ─────────────────────────────────────────────────────────────────────────────

def load_historical(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col="date", parse_dates=True)

def load_future(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col="date", parse_dates=True)
    if "predicted_price" in df.columns and "predicted_index" not in df.columns:
        df = df.rename(columns={"predicted_price": "predicted_index"})
    return df

def load_feature_importance(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower()
    if "feature" not in df.columns: df.columns = ["feature", "importance"]
    return df.sort_values("importance", ascending=False).head(20)

def load_metadata(path: str) -> dict:
    with open(path) as f: return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# CHART HELPERS & LAYOUT (Light Theme)
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_shapes(dates: pd.DatetimeIndex, probs: np.ndarray, threshold: float = 0.5) -> list:
    labels = (probs > threshold).astype(int)
    shapes, in_block, t0 = [], False, None
    for d, lbl in zip(dates, labels):
        if lbl == 1 and not in_block:
            in_block, t0 = True, d
        elif lbl == 0 and in_block:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(d), y0=0, y1=1,
                               fillcolor=C_REG1, line_width=0, layer="below"))
            in_block = False
    if in_block:
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(dates[-1]), y0=0, y1=1,
                           fillcolor=C_REG1, line_width=0, layer="below"))
    return shapes

def _layout(title: str, y_title: str = "Price", height: int = 460) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=16, color="#111"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", bordercolor="#CCC", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=55, r=20, t=55, b=40),
    )

def _add_shock_vlines(fig: go.Figure, date_min: pd.Timestamp):
    for ds, label in SHOCK_ANNOTATIONS.items():
        d = pd.Timestamp(ds)
        if d >= date_min:
            fig.add_vline(x=d.timestamp()*1000, line=dict(color="rgba(200,0,0,0.5)", width=1.5, dash="dot"),
                          annotation_text=label, annotation_font=dict(size=9, color="#B71C1C"), annotation_position="top")

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_price_comparison(hist: pd.DataFrame, future: pd.DataFrame, display_window: int, price_mode: str, show_regime: bool, show_market: bool) -> go.Figure:
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    if price_mode == "real":
        actual_vals = h["real_price"] if "real_price" in h else h["actual"]
        hybrid_vals = h["real_price"] if "real_price" in h else h.get("hybrid_prediction", h["lgb_prediction"])
        y_title, future_col = "Price (Rs/Kg)", "real_price"
    else:
        actual_vals = h["actual"]
        hybrid_vals = h["hybrid_prediction"] if "hybrid_prediction" in h.columns else h["lgb_prediction"]
        y_title, future_col = "Price Index", "predicted_index"

    fig = go.Figure()

    if show_regime and "regime_probability" in h.columns:
        for s in build_regime_shapes(h.index, h["regime_probability"].values): fig.add_shape(**s)

    fig.add_trace(go.Scatter(x=h.index, y=actual_vals, name="Model Index/Base", mode="lines", line=dict(color=C_HYBRID, width=3, dash="solid")))
    
    if show_market and "market_price" in h.columns:
        # Added the Market Line to the main tab as requested
        fig.add_trace(go.Scatter(x=h.index, y=h["market_price"], name="Actual Market Price", mode="lines", line=dict(color=C_MARKET, width=2.5)))

    if not future.empty and future_col in future.columns:
        fp = future[future_col]
        fig.add_trace(go.Scatter(x=future.index, y=fp, name="Future Forecast", mode="lines", line=dict(color=C_FUTURE, width=3)))

    _add_shock_vlines(fig, h.index[0])
    fig.update_layout(**_layout("Mn Briquette Hybrid Price Forecast", y_title=y_title, height=470))
    return fig


def chart_dual_axis(hist: pd.DataFrame, future: pd.DataFrame, display_window: int) -> go.Figure:
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Index -> Solid Blue
    fig.add_trace(go.Scatter(x=h.index, y=h["hybrid_prediction"] if "hybrid_prediction" in h else h["lgb_prediction"],
                             name="Historical Index", mode="lines", line=dict(color=C_HYBRID, width=3, dash="solid")), secondary_y=False)

    # Price -> Dashed Orange
    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(x=h.index, y=h["real_price"], name="Historical Price", mode="lines", 
                                 line=dict(color=C_ACTUAL, width=2, dash="dash")), secondary_y=True)

    if not future.empty:
        # Future Index -> Solid Light Blue
        if "predicted_index" in future.columns:
            fig.add_trace(go.Scatter(x=future.index, y=future["predicted_index"], name="Future Index", mode="lines", 
                                     line=dict(color="#64B5F6", width=3, dash="solid")), secondary_y=False)
        # Future Price -> Dashed Red/Orange
        if "real_price" in future.columns:
            fig.add_trace(go.Scatter(x=future.index, y=future["real_price"], name="Future Price", mode="lines", 
                                     line=dict(color="#FF5722", width=2, dash="dash")), secondary_y=True)

    fig.update_layout(**_layout("Index vs Price — Dual Axis Separation", "Price Index", 400))
    fig.update_yaxes(title_text="Price Index (Solid Lines)", secondary_y=False, showgrid=True, gridcolor=C_GRID)
    fig.update_yaxes(title_text="Price Rs/Kg (Dashed Lines)", secondary_y=True, showgrid=False)
    return fig


def chart_market_comparison(hist: pd.DataFrame) -> go.Figure | None:
    if "market_price" not in hist.columns: return None
    common = hist.dropna(subset=["market_price"]).copy()
    if common.empty: return None

    fig = go.Figure()
    # Market Price -> Bright Magenta (was black)
    fig.add_trace(go.Scatter(x=common.index, y=common["market_price"], name="Actual Market Price", mode="lines", line=dict(color=C_MARKET, width=3)))
    # Model Predicted Price -> Blue Dash
    if "real_price" in common.columns:
        fig.add_trace(go.Scatter(x=common.index, y=common["real_price"], name="Model Calibrated Price", mode="lines", line=dict(color=C_HYBRID, width=2.5, dash="dash")))

    fig.update_layout(**_layout("Market Comparison: Real vs Predicted (Overlapping Window)", y_title="Rs/Kg", height=380))
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE CHARTS (Bar & Donut)
# ═════════════════════════════════════════════════════════════════════════════

def chart_feature_bars(fi: pd.DataFrame) -> go.Figure:
    top = fi.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker=dict(color="#3F51B5", opacity=0.85)))
    fig.update_layout(**_layout("Top Driver Importance (Raw Score)", "Score", 400))
    fig.update_layout(yaxis=dict(tickfont=dict(size=10), showgrid=False), margin=dict(l=150, r=20, t=55, b=40))
    return fig

def chart_feature_donut(fi: pd.DataFrame) -> go.Figure:
    # Group minor features into "Other"
    top = fi.head(7).copy()
    other_val = fi.iloc[7:]["importance"].sum()
    if other_val > 0:
        top = pd.concat([top, pd.DataFrame([{"feature": "Other Variables", "importance": other_val}])])
    
    # Calculate Percentage
    top["percent"] = (top["importance"] / top["importance"].sum()) * 100
    
    fig = go.Figure(data=[go.Pie(labels=top["feature"], values=top["percent"], hole=0.5, 
                                 textinfo='label+percent', marker=dict(colors=["#1E88E5", "#43A047", "#E53935", "#FB8C00", "#8E24AA", "#00ACC1", "#7CB342", "#9E9E9E"]))])
    fig.update_layout(title=dict(text="% Dependence of Mn Briquette Price on Drivers", font=dict(size=16, color="#111"), x=0.5),
                      template="plotly_white", paper_bgcolor="white", height=400, showlegend=False)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR & MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style='padding:12px 0 6px;'>
      <h1 style='font-size:25px;margin:0;color:#111;'>
        📈 Mn Briquette Hybrid Price Forecasting &nbsp; <span style='font-size:14px;color:#666;font-weight:400;'>v4.0</span>
      </h1>
      <p style='color:#555;font-size:13px;margin:3px 0 0;'>
        Structural Break Modeling | Rolling Dynamics | Exogenous Driver Substitution
      </p>
    </div>
    """, unsafe_allow_html=True)

    meta_path = os.path.join(OUTPUT_DIR, "model_metadata.json")
    meta = load_metadata(meta_path) if os.path.exists(meta_path) else {}

    with st.sidebar:
        st.markdown("## ⚙️ Controls")
        st.divider()
        display_window = st.slider("History to show (weeks)", 52, 520, 260, step=26)
        horizon_label  = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()), index=2)
        price_mode     = st.radio("Display prices as", ["Real price (Rs/Kg)", "Index value"], index=0)
        show_regime    = st.checkbox("Show regime shading", value=True)
        show_market    = st.checkbox("Show market price (Excel)", value=True)
        show_dual      = st.checkbox("Show dual-axis chart", value=False)
        st.divider()
        st.caption("Target: Mn Briquette (97%)\nModel: LightGBM + Regime Tracking")

    # Load Data
    hist_path   = os.path.join(OUTPUT_DIR, "historical_predictions.csv")
    future_path = os.path.join(OUTPUT_DIR, "future_forecast.csv")
    fi_path     = os.path.join(OUTPUT_DIR, "feature_importance.csv")

    if not os.path.exists(hist_path) or not os.path.exists(future_path):
        st.error("⛔ Missing pipeline outputs. Run `pipeline.py` first.")
        st.stop()

    hist   = load_historical(hist_path)
    future = load_future(future_path).iloc[:HORIZON_OPTIONS[horizon_label]]
    fi     = load_feature_importance(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    last_real = float(hist["real_price"].iloc[-1]) if "real_price" in hist.columns else 0.0
    end_real  = float(future["real_price"].iloc[-1]) if "real_price" in future.columns else 0.0
    c1.metric("Last Actual (Rs/Kg)",  f"₹{last_real:.2f}")
    c2.metric("End Forecast (Rs/Kg)", f"₹{end_real:.2f}", delta=f"{len(future)} wks ahead", delta_color="off")
    if "market_price" in hist.columns:
        last_mkt = hist["market_price"].dropna()
        if not last_mkt.empty:
            c3.metric("Last Excel Market Price", f"₹{last_mkt.iloc[-1]:.2f}")

    tab1, tab2, tab3 = st.tabs(["📈 Price Forecast", "📊 Market Comparison", "🔀 Regime & Drivers"])

    with tab1:
        st.plotly_chart(chart_price_comparison(hist, future, display_window, "real" if "Real" in price_mode else "index", show_regime, show_market), use_container_width=True)
        if show_dual:
            st.plotly_chart(chart_dual_axis(hist, future, display_window), use_container_width=True)

    with tab2:
        mkt_chart = chart_market_comparison(hist)
        if mkt_chart: st.plotly_chart(mkt_chart, use_container_width=True)
        else: st.info("📂 No market price data found. Ensure market_prices.xlsx is loaded in pipeline.")

    with tab3:
        # Regime section
        if "regime_probability" in hist.columns:
            st.markdown("### Market Regime State")
            cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
            h = hist[hist.index >= cutoff]
            fig_r = go.Figure(go.Scatter(x=h.index, y=h["regime_probability"], name="P(Supply Squeeze)", fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.2)", line=dict(color="#D32F2F")))
            fig_r.update_layout(**_layout("P(Supply Squeeze / Ore Deficit)", "Probability", 250))
            st.plotly_chart(fig_r, use_container_width=True)

        st.divider()
        st.markdown("### Fundamental Driver Dependence")
        if not fi.empty:
            fc1, fc2 = st.columns(2)
            with fc1: st.plotly_chart(chart_feature_bars(fi), use_container_width=True)
            with fc2: st.plotly_chart(chart_feature_donut(fi), use_container_width=True)

if __name__ == "__main__":
    main()