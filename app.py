"""
Mn Briquette Hybrid Forecasting — Streamlit Dashboard  ·  v4.2
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
    page_title="Mn Briquette Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light Theme Colors
C_ACTUAL   = "#FF9800" # Orange for Prices
C_HYBRID   = "#2196F3" # Blue for Index
C_FUTURE   = "#4CAF50" # Green for Future
C_MARKET   = "#9C27B0" # Deep Magenta for Market Data
C_REG1     = "rgba(244, 67, 54, 0.15)" # Light Red for Squeeze
C_GRID     = "#EEEEEE" # Light Grey Grid
C_TEXT     = "#333333" # Dark Grey Text
C_CI       = "rgba(76, 175, 80, 0.12)" # Light Green for CI

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
# DATA LOADERS
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
# LAYOUT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_shapes(dates: pd.DatetimeIndex, probs: np.ndarray, threshold: float = 0.5) -> list:
    labels = (probs > threshold).astype(int)
    shapes, in_block, t0 = [], False, None
    for d, lbl in zip(dates, labels):
        if lbl == 1 and not in_block:
            in_block, t0 = True, d
        elif lbl == 0 and in_block:
            shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(d), y0=0, y1=1, fillcolor=C_REG1, line_width=0, layer="below"))
            in_block = False
    if in_block:
        shapes.append(dict(type="rect", xref="x", yref="paper", x0=str(t0), x1=str(dates[-1]), y0=0, y1=1, fillcolor=C_REG1, line_width=0, layer="below"))
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
            fig.add_vline(x=d.timestamp()*1000, line=dict(color="rgba(200,0,0,0.5)", width=1.5, dash="dot"), annotation_text=label, annotation_font=dict(size=9, color="#B71C1C"), annotation_position="top")

def _confidence_band(fig: go.Figure, dates, values: np.ndarray):
    n = len(values)
    idx = np.arange(1, n+1)
    sigma = np.std(values) * 0.015 * idx
    upper = values + 1.96*sigma
    lower = values - 1.96*sigma
    fig.add_trace(go.Scatter(x=list(dates)+list(dates[::-1]), y=list(upper)+list(lower[::-1]), fill="toself", fillcolor=C_CI, line=dict(width=0), name="95% CI", showlegend=True, hoverinfo="skip"))

# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def chart_price_comparison(hist: pd.DataFrame, future: pd.DataFrame, display_window: int, price_mode: str, show_regime: bool, show_market: bool) -> go.Figure:
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()

    if price_mode == "real":
        actual_vals = h.get("real_price", h["actual"])
        y_title, future_col, fmt = "Price (Rs/Kg)", "real_price", "₹%{y:.2f}"
    else:
        actual_vals = h["actual"]
        y_title, future_col, fmt = "Price Index", "predicted_index", "%{y:.2f}"

    fig = go.Figure()

    if show_regime and "regime_probability" in h.columns:
        for s in build_regime_shapes(h.index, h["regime_probability"].values): fig.add_shape(**s)

    fig.add_trace(go.Scatter(x=h.index, y=actual_vals, name="Index Price", mode="lines", line=dict(color=C_HYBRID, width=3),
                             hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: " + fmt + "<extra></extra>"))
    
    if show_market and "market_price" in h.columns:
        fig.add_trace(go.Scatter(x=h.index, y=h["market_price"], name="Market Price", mode="lines", line=dict(color=C_MARKET, width=2.5),
                                 hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Market Price: " + fmt + "<extra></extra>"))

    if not future.empty and future_col in future.columns:
        fp = future[future_col]
        _confidence_band(fig, future.index, fp.values)
        fig.add_trace(go.Scatter(x=future.index, y=fp, name="Future Forecast Price", mode="lines", line=dict(color=C_FUTURE, width=3),
                                 hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Forecast Price: " + fmt + "<extra></extra>"))

    _add_shock_vlines(fig, h.index[0])
    fig.update_layout(**_layout("Mn Briquette Price Forecast", y_title=y_title, height=470))
    return fig


def chart_dual_axis(hist: pd.DataFrame, future: pd.DataFrame, display_window: int) -> go.Figure:
    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(x=h.index, y=h.get("hybrid_prediction", h["lgb_prediction"]), name="Historical Index Price", mode="lines", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Historical Index Price: %{y:.2f}<extra></extra>"), secondary_y=False)
    
    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(x=h.index, y=h["real_price"], name="Historical Price", mode="lines", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Historical Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

    if not future.empty:
        if "predicted_index" in future.columns:
            fig.add_trace(go.Scatter(x=future.index, y=future["predicted_index"], name="Future Index Price", mode="lines", line=dict(color="#64B5F6", width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Index Price: %{y:.2f}<extra></extra>"), secondary_y=False)
        if "real_price" in future.columns:
            fig.add_trace(go.Scatter(x=future.index, y=future["real_price"], name="Future Price", mode="lines", line=dict(color="#FF5722", width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Price: ₹%{y:.2f}<extra></extra>"), secondary_y=True)

    fig.update_layout(**_layout("Index vs Price — Dual Axis Separation", "Price Index", 400))
    fig.update_yaxes(title_text="Price Index (Solid Lines)", secondary_y=False, showgrid=True, gridcolor=C_GRID)
    fig.update_yaxes(title_text="Price Rs/Kg (Dashed Lines)", secondary_y=True, showgrid=False)
    return fig


def chart_market_comparison(hist: pd.DataFrame) -> go.Figure | None:
    if "market_price" not in hist.columns: return None
    common = hist.dropna(subset=["market_price"]).copy()
    if common.empty: return None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=common.index, y=common["market_price"], name="Market Price", mode="lines", line=dict(color=C_MARKET, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Market Price: ₹%{y:.2f}<extra></extra>"))
    if "real_price" in common.columns:
        fig.add_trace(go.Scatter(x=common.index, y=common["real_price"], name="Index Price", mode="lines", line=dict(color=C_HYBRID, width=2.5, dash="dash"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: ₹%{y:.2f}<extra></extra>"))

    fig.update_layout(**_layout("Market Comparison: Real vs Predicted", y_title="Rs/Kg", height=380))
    return fig


def chart_feature_bars(fi: pd.DataFrame) -> go.Figure:
    top = fi.head(15).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker=dict(color="#3F51B5", opacity=0.85), hovertemplate="Feature: %{y}<br>Score: %{x}<extra></extra>"))
    fig.update_layout(**_layout("Top Driver Importance", "Score", 400))
    fig.update_layout(yaxis=dict(tickfont=dict(size=10), showgrid=False), margin=dict(l=150, r=20, t=55, b=40))
    return fig

def chart_feature_donut(fi: pd.DataFrame) -> go.Figure:
    top = fi.head(7).copy()
    other_val = fi.iloc[7:]["importance"].sum()
    if other_val > 0: top = pd.concat([top, pd.DataFrame([{"feature": "Other Variables", "importance": other_val}])])
    top["percent"] = (top["importance"] / top["importance"].sum()) * 100
    fig = go.Figure(data=[go.Pie(labels=top["feature"], values=top["percent"], hole=0.5, textinfo='label+percent', marker=dict(colors=["#1E88E5", "#43A047", "#E53935", "#FB8C00", "#8E24AA", "#00ACC1", "#7CB342", "#9E9E9E"]), hovertemplate="%{label}<br>Dependence: %{percent:.1f}%<extra></extra>")])
    fig.update_layout(title=dict(text="% Dependence of Mn Briquette Price on Drivers", font=dict(size=16, color="#111"), x=0.5), template="plotly_white", paper_bgcolor="white", height=400, showlegend=False)
    return fig

# ═════════════════════════════════════════════════════════════════════════════
# STEEL VALUE CALCULATOR
# ═════════════════════════════════════════════════════════════════════════════

def render_steel_calculator(hist: pd.DataFrame, future: pd.DataFrame, display_window: int):
    st.markdown("### 🏭 Steel Production Value Calculator")
    st.caption("Computes the **economic value of Mn** per kg of effective manganese delivered. Updates dynamically as model prices change.")

    col_mn, col_rec, col_spacer = st.columns([1, 1, 3])
    with col_mn: mn_pct = st.slider("Mn content (%)", min_value=1.0, max_value=99.0, value=97.0, step=0.5)
    with col_rec: rec_pct = st.slider("Recovery (%)", min_value=50.0, max_value=100.0, value=92.0, step=0.5)

    eff_mn_frac = (mn_pct / 100.0) * (rec_pct / 100.0)
    eff_mn_pct  = eff_mn_frac * 100.0

    latest_real = float(hist["real_price"].iloc[-1]) if "real_price" in hist.columns else 0.0
    real_mn_value = latest_real / eff_mn_frac if eff_mn_frac > 0 else 0.0

    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Mn Content", f"{mn_pct:.1f}%")
    kc2.metric("Recovery", f"{rec_pct:.1f}%")
    kc3.metric("Effective Mn", f"{eff_mn_pct:.2f}%")
    kc4.metric("Real Mn Value", f"₹{real_mn_value:.2f}/Kg", help="Rs per kg of effective Mn delivered")

    cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
    h = hist[hist.index >= cutoff].copy()
    fut = future.copy()

    h["mn_value"] = h["real_price"] / eff_mn_frac if "real_price" in h.columns else np.nan
    fut["mn_value"] = fut["real_price"] / eff_mn_frac if "real_price" in fut.columns else np.nan

    fig = go.Figure()
    if "real_price" in h.columns:
        fig.add_trace(go.Scatter(x=h.index, y=h["real_price"], name="Index Price (Rs/Kg)", mode="lines", line=dict(color=C_HYBRID, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Index Price: ₹%{y:.2f}<extra></extra>"))
    if "real_price" in fut.columns:
        fig.add_trace(go.Scatter(x=fut.index, y=fut["real_price"], name="Future Index Price", mode="lines", line=dict(color=C_FUTURE, width=3), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Index Price: ₹%{y:.2f}<extra></extra>"))
    if not h["mn_value"].isna().all():
        fig.add_trace(go.Scatter(x=h.index, y=h["mn_value"], name="Mn Value (Rs/Kg Mn)", mode="lines", line=dict(color=C_ACTUAL, width=2.5, dash="dash"), yaxis="y2", hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Mn Value: ₹%{y:.2f}<extra></extra>"))
    if not fut["mn_value"].isna().all():
        fig.add_trace(go.Scatter(x=fut.index, y=fut["mn_value"], name="Future Mn Value", mode="lines", line=dict(color="#FF5722", width=2.5, dash="dash"), yaxis="y2", hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>Future Mn Value: ₹%{y:.2f}<extra></extra>"))

    if not future.empty:
        fig.add_vline(x=hist.index[-1].timestamp()*1000, line=dict(color="rgba(0,0,0,0.3)", width=1, dash="dot"), annotation_text="Today")

    fig.update_layout(**_layout(f"Price vs Mn Value (Mn={mn_pct:.1f}%, Rec={rec_pct:.1f}%)", "Price (Rs/Kg)", 430))
    fig.update_layout(yaxis2=dict(title="Mn Value (Rs/Kg Mn)", overlaying="y", side="right", showgrid=False, matches="y"))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Steel Value Table (sampled — every 4 weeks)"):
        all_dates = list(h.index[::4]) + list(fut.index[::4])
        all_real  = list(h["real_price"].iloc[::4]) + list(fut["real_price"].iloc[::4])
        all_mn    = list(h["mn_value"].iloc[::4]) + list(fut["mn_value"].iloc[::4])
        tbl = pd.DataFrame({
            "Date": [d.strftime("%Y-%m-%d") for d in all_dates],
            "Index Price Rs/Kg": all_real, "Mn Value Rs/Kg": all_mn,
            "Period": ["Historical"]*len(h.index[::4]) + ["Forecast"]*len(fut.index[::4]),
        })
        st.dataframe(tbl.style.format({"Index Price Rs/Kg": "{:.2f}", "Mn Value Rs/Kg": "{:.2f}"}).map(lambda v: "color: #4CAF50" if v == "Forecast" else "", subset=["Period"]), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style='padding:12px 0 6px;'>
      <h1 style='font-size:25px;margin:0;color:#111;'>📈 Mn Briquette Price Forecasting &nbsp; <span style='font-size:14px;color:#666;'>v4.2</span></h1>
      <p style='color:#555;font-size:13px;margin:3px 0 0;'>Structural Growth Modeling | Rolling Dynamics | Steel Value Optimizer</p>
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
        show_market    = st.checkbox("Show market price", value=True)
        show_dual      = st.checkbox("Show dual-axis chart", value=False)
        
        st.divider()
        st.caption("Target: Mn Briquette (97%)\nModel: LightGBM + Regime Tracking")

    hist_path, future_path, fi_path = os.path.join(OUTPUT_DIR, "historical_predictions.csv"), os.path.join(OUTPUT_DIR, "future_forecast.csv"), os.path.join(OUTPUT_DIR, "feature_importance.csv")

    if not os.path.exists(hist_path) or not os.path.exists(future_path):
        st.error("⛔ Missing pipeline outputs. Run `pipeline.py` first.")
        st.stop()

    hist   = load_historical(hist_path)
    future = load_future(future_path).iloc[:HORIZON_OPTIONS[horizon_label]]
    fi     = load_feature_importance(fi_path) if os.path.exists(fi_path) else pd.DataFrame()

    pred_col = "hybrid_prediction" if "hybrid_prediction" in hist else "lgb_prediction"
    mape = float(np.mean(np.abs((hist["actual"] - hist[pred_col]) / (hist["actual"] + 1e-9))) * 100)
    cal = meta.get("calibration", {})

    # Main Top KPIs (Fixed Format)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    last_idx = float(hist["actual"].iloc[-1])
    last_real = float(hist["real_price"].iloc[-1]) if "real_price" in hist.columns else 0.0
    nxt_real  = float(future["real_price"].iloc[0]) if "real_price" in future.columns else 0.0
    end_real  = float(future["real_price"].iloc[-1]) if "real_price" in future.columns else 0.0
    pct_chg   = ((nxt_real - last_real) / last_real) * 100 if last_real else 0
    
    c1.metric("Last Index", f"{last_idx:.2f}")
    c2.metric("Last Actual (Rs/Kg)",  f"₹{last_real:.2f}")
    c3.metric("Next-Wk Forecast", f"₹{nxt_real:.2f}", delta=f"{pct_chg:+.1f}%")
    c4.metric("End Forecast", f"₹{end_real:.2f}", delta=f"{len(future)} wks ahead", delta_color="off")
    c5.metric("In-Sample MAPE", f"{mape:.2f}%")
    c6.metric("Scaling Factor", f"{cal.get('scaling_factor', 0):.4f}", help="Anchor Price: 2024-01-02 = ₹154.46585/Kg")

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Price Forecast", "📊 Market Comparison", "🔀 Regime & Drivers", "🏭 Steel Calculator"])

    with tab1:
        st.plotly_chart(chart_price_comparison(hist, future, display_window, "real" if "Real" in price_mode else "index", show_regime, show_market), use_container_width=True)
        if show_dual: st.plotly_chart(chart_dual_axis(hist, future, display_window), use_container_width=True)
        
        with st.expander("🔮 Future Forecast Detail"):
            disp_fut = future.copy().reset_index()
            disp_fut["date"] = disp_fut["date"].dt.strftime("%Y-%m-%d")
            if "real_price" in disp_fut.columns: disp_fut["real_price_fmt"] = disp_fut["real_price"].apply(lambda x: f"₹{x:.2f}/Kg")
            if "regime_probability" in disp_fut.columns: disp_fut["regime"] = disp_fut["regime_probability"].apply(lambda p: "🔴 Squeeze" if p > 0.5 else "🟢 Normal")
            cols_show = [c for c in ["date","predicted_index","real_price_fmt","regime_probability","regime"] if c in disp_fut.columns]
            st.dataframe(disp_fut[cols_show], use_container_width=True, hide_index=True)

    with tab2:
        mkt_chart = chart_market_comparison(hist)
        if mkt_chart:
            st.plotly_chart(mkt_chart, use_container_width=True)
            common = hist[["real_price", "market_price"]].dropna()
            if not common.empty:
                err = common["real_price"] - common["market_price"]
                ec1, ec2, ec3 = st.columns(3)
                ec1.metric("RMSE vs Market", f"₹{float(np.sqrt((err**2).mean())):.2f}")
                ec2.metric("MAE vs Market",  f"₹{float(err.abs().mean()):.2f}")
                ec3.metric("MAPE vs Market", f"{float((err.abs() / common['market_price']).mean() * 100):.2f}%")
                st.markdown("##### 📋 Error Comparison Table")
                st.dataframe(pd.DataFrame({"Date": common.index.strftime("%Y-%m-%d"), "Market Price": common["market_price"].round(2), "Model Price": common["real_price"].round(2), "Error %": ((err / common["market_price"]) * 100).round(2).astype(str) + "%"}), use_container_width=True, hide_index=True)
        else: st.info("📂 No market price data found.")
        
        with st.expander("🧾 Model Metadata & Calibration Details"): st.json(meta)

    with tab3:
        if "regime_probability" in hist.columns:
            st.markdown("### Market Regime State")
            cutoff = hist.index[-1] - pd.DateOffset(weeks=display_window)
            h = hist[hist.index >= cutoff]
            fig_r = go.Figure(go.Scatter(x=h.index, y=h["regime_probability"], name="P(Supply Squeeze)", fill="tozeroy", fillcolor="rgba(244, 67, 54, 0.2)", line=dict(color="#D32F2F"), hovertemplate="<b>%{x|%B %Y}</b><br>%{x|%Y-%m-%d}<br>P(Squeeze): %{y:.2f}<extra></extra>"))
            fig_r.update_layout(**_layout("P(Supply Squeeze / Ore Deficit)", "Probability", 250))
            st.plotly_chart(fig_r, use_container_width=True)
            
            r1c, r2c = st.columns(2)
            pct1 = float((hist["regime_probability"] > 0.5).mean() * 100)
            r1c.metric("🔴 Regime 1 — Supply Squeeze", f"{pct1:.1f}% of historical time")
            r2c.metric("🟢 Regime 0 — Oversupply / Normal", f"{(100 - pct1):.1f}% of historical time")

        st.divider()
        st.markdown("### Fundamental Driver Dependence")
        if not fi.empty:
            fc1, fc2 = st.columns(2)
            with fc1: st.plotly_chart(chart_feature_bars(fi), use_container_width=True)
            with fc2: st.plotly_chart(chart_feature_donut(fi), use_container_width=True)
            
        with st.expander("🧾 Model Metadata & Calibration Details"): st.json(meta)

    with tab4:
        render_steel_calculator(hist, future, display_window)
        with st.expander("🧾 Model Metadata & Calibration Details"): st.json(meta)

if __name__ == "__main__":
    main()