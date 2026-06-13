"""
================================================================================
Phase 5: Streamlit "What-If" Simulator Dashboard
================================================================================

Interactive frontend for the Prescriptive Demand Forecasting Engine.
Combines historical data, TFT baseline quantiles, and Causal DML elasticities
to simulate the impact of price interventions on demand and revenue.

To run:
    streamlit run app.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Import our Phase 4 Integration Engine
from integration_eval import PrescriptiveEngine

# ==============================================================================
# CONFIG & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Prescriptive Demand Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished metric cards
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
    }
    .metric-title {
        color: #6c757d;
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #212529;
    }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# DATA LOADING (CACHED)
# ==============================================================================
@st.cache_data
def load_dashboard_data():
    """
    Loads baseline forecasts and elasticities. 
    In production, this reads the parquet files dumped by the pipeline.
    Here, we generate high-quality synthetic data for a seamless demo.
    """
    np.random.seed(42)
    
    items = [f"ITEM_{str(i).zfill(3)}" for i in range(1, 6)]
    stores = ["STORE_1", "STORE_2"]
    
    # 1. Historical Sales (Past 30 days)
    history = []
    for store in stores:
        for item in items:
            base_sales = np.random.uniform(50, 200)
            for day in range(-30, 0):
                # Add some seasonality and noise
                sales = base_sales + 20 * np.sin(day / 7.0) + np.random.normal(0, 5)
                history.append({
                    "store_id": store,
                    "item_id": item,
                    "day": day,
                    "sales": max(0, sales),
                    "price": 10.0 + np.random.normal(0, 1) # ~ $10 base price
                })
    df_hist = pd.DataFrame(history)

    # 2. TFT Baseline Predictions (Next 14 days)
    predictions = []
    for store in stores:
        for item in items:
            last_sales = df_hist[(df_hist["store_id"] == store) & (df_hist["item_id"] == item)]["sales"].iloc[-1]
            for step in range(1, 15):
                # Baseline median
                p50 = last_sales + np.random.normal(0, 2)
                # Uncertainty widens over time
                spread = 10 + (step * 1.5)
                
                predictions.append({
                    "store_id": store,
                    "item_id": item,
                    "horizon_step": step, # Positive means future
                    "p10": max(0, p50 - spread),
                    "p50": max(0, p50),
                    "p90": max(0, p50 + spread),
                    "base_price": 10.0
                })
    df_pred = pd.DataFrame(predictions)

    # 3. Causal Elasticities (tau)
    # Different elasticities per item
    elasticities = []
    for item in items:
        # e.g. electronics highly elastic (-2.5), staples less elastic (-0.5)
        tau = np.random.uniform(-3.0, -0.5)
        for store in stores:
            elasticities.append({
                "store_id": store,
                "item_id": item,
                "cat_id": "CAT_A" if "1" in item or "2" in item else "CAT_B",
                "tau": tau + np.random.normal(0, 0.2) # Slight store variation
            })
    df_causal = pd.DataFrame(elasticities)

    return df_hist, df_pred, df_causal


df_hist, df_pred, df_causal = load_dashboard_data()


# ==============================================================================
# UI COMPONENTS
# ==============================================================================

st.title("📊 Prescriptive Demand Simulator")
st.markdown("""
Welcome to the Quant Risk & Pricing Dashboard. This tool fuses **Temporal Fusion Transformer (TFT)** probabilistic 
forecasts with **Double Machine Learning (DML)** causal elasticities. Adjust the price intervention slider to simulate 
the causal impact on future demand volume and revenue.
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎛️ Simulation Controls")
    
    selected_store = st.selectbox("Select Store:", df_pred["store_id"].unique())
    selected_item = st.selectbox("Select Product:", df_pred["item_id"].unique())
    
    st.markdown("---")
    
    st.subheader("Pricing Strategy")
    st.markdown("Simulate a price change relative to the baseline.")
    delta_p_pct = st.slider(
        "Price Intervention (\u0394P)",
        min_value=-30,
        max_value=30,
        value=0,
        step=1,
        format="%d%%"
    )
    delta_p = delta_p_pct / 100.0
    
    st.markdown("---")
    
    st.subheader("Inventory Strategy")
    risk_tolerance = st.radio(
        "Optimization Target (Safety Stock):",
        options=["Median Expected Demand (p50)", "Conservative Upside (p90)"],
        index=0,
        help="Choose p50 for standard inventory optimization, or p90 to minimize stockouts during demand spikes."
    )
    target_quantile_col = "final_p50" if "p50" in risk_tolerance else "final_p90"


# ==============================================================================
# PRESCRIPTIVE INTEGRATION (Reactive)
# ==============================================================================

# Filter baseline data for the selected item/store
filt_pred = df_pred[(df_pred["store_id"] == selected_store) & (df_pred["item_id"] == selected_item)].copy()
filt_hist = df_hist[(df_hist["store_id"] == selected_store) & (df_hist["item_id"] == selected_item)].copy()

# Filter causal data
item_elasticity_df = df_causal[(df_causal["store_id"] == selected_store) & (df_causal["item_id"] == selected_item)]
item_tau = item_elasticity_df["tau"].values[0] if not item_elasticity_df.empty else 0.0

# Apply the Integration Engine mathematically (instantaneous update)
df_prescriptive = PrescriptiveEngine.generate_prescriptive_forecast(
    df_tft_predictions=filt_pred,
    df_causal_elasticities=df_causal, # Pass full causal df; the engine maps it
    delta_p=delta_p
)

# Sort strictly by time for plotting
filt_hist = filt_hist.sort_values("day")
df_prescriptive = df_prescriptive.sort_values("horizon_step")

# Revenue calculations
base_price = filt_pred["base_price"].iloc[0]
new_price = base_price * (1.0 + delta_p)

# Expected volumes (sum over 14 days)
base_volume = filt_pred["p50"].sum()
new_volume = df_prescriptive["final_p50"].sum()

# Expected revenues
base_revenue = base_volume * base_price
new_revenue = new_volume * new_price
revenue_impact = new_revenue - base_revenue

# Stockout Risk: If the downside volume falls below a critical threshold (e.g. 10 units a day)
min_downside = df_prescriptive["final_p10"].min()
stockout_risk = "High" if min_downside < 10 else ("Medium" if min_downside < 30 else "Low")
risk_color = "🔴" if stockout_risk == "High" else ("🟡" if stockout_risk == "Medium" else "🟢")


# ==============================================================================
# METRICS ROW
# ==============================================================================

st.markdown("### 📈 Operational Insights")
col1, col2, col3, col4 = st.columns(4)

with col1:
    elasticity_label = "Highly Elastic" if item_tau < -1.5 else "Inelastic"
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Price Elasticity (τ)</div>
            <div class="metric-value">{item_tau:.2f}</div>
            <div style="color: #6c757d; font-size: 0.8rem;">{elasticity_label}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Projected Volume (14d)</div>
            <div class="metric-value">{int(new_volume):,} units</div>
            <div style="color: {'#28a745' if new_volume >= base_volume else '#dc3545'}; font-size: 0.8rem;">
                {int(new_volume - base_volume):+} vs Baseline
            </div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Revenue Impact (14d)</div>
            <div class="metric-value">${abs(revenue_impact):,.0f}</div>
            <div style="color: {'#28a745' if revenue_impact >= 0 else '#dc3545'}; font-size: 0.8rem;">
                {"Gain" if revenue_impact >= 0 else "Loss"} vs Baseline
            </div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Stockout Risk (p10)</div>
            <div class="metric-value">{risk_color} {stockout_risk}</div>
            <div style="color: #6c757d; font-size: 0.8rem;">Downside floor: {int(min_downside)} units/day</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ==============================================================================
# MAIN VISUALIZATION (PLOTLY FAN CHART)
# ==============================================================================
st.markdown("### 📊 Time-Series Probabilistic Fan Chart")

fig = go.Figure()

# 1. Plot Historical Sales
fig.add_trace(go.Scatter(
    x=filt_hist["day"],
    y=filt_hist["sales"],
    mode="lines+markers",
    name="Historical Sales",
    line=dict(color="#1f77b4", width=2),
    marker=dict(size=4)
))

# 2. Plot Uncertainty Band (p10 to p90)
# We draw the p90 line, then the p10 line with 'fill="tonexty"' to shade between them.
# x-axis for forecast starts at day 0 (stitching history to forecast)
forecast_x = np.concatenate([[0], df_prescriptive["horizon_step"].values])

# Stitch the last historical point to the forecast arrays
last_hist_sales = filt_hist["sales"].iloc[-1]
p90_y = np.concatenate([[last_hist_sales], df_prescriptive["final_p90"].values])
p10_y = np.concatenate([[last_hist_sales], df_prescriptive["final_p10"].values])
p50_y = np.concatenate([[last_hist_sales], df_prescriptive["final_p50"].values])

target_y = np.concatenate([[last_hist_sales], df_prescriptive[target_quantile_col].values])

# Upper Bound (p90)
fig.add_trace(go.Scatter(
    x=forecast_x,
    y=p90_y,
    mode="lines",
    line=dict(width=0),
    showlegend=False,
    hoverinfo="skip"
))

# Lower Bound (p10) + Fill
fig.add_trace(go.Scatter(
    x=forecast_x,
    y=p10_y,
    mode="lines",
    fill="tonexty", # Fills to the previously added trace (p90)
    fillcolor="rgba(255, 127, 14, 0.2)",
    line=dict(width=0),
    name="80% Prediction Interval (p10 - p90)"
))

# 3. Plot Median Forecast (p50)
fig.add_trace(go.Scatter(
    x=forecast_x,
    y=p50_y,
    mode="lines",
    name="Median Forecast (p50)",
    line=dict(color="#ff7f0e", width=3, dash="dash")
))

# 4. Plot Optimized Target (if different from p50)
if target_quantile_col != "final_p50":
    fig.add_trace(go.Scatter(
        x=forecast_x,
        y=target_y,
        mode="lines",
        name="Target Safety Stock (p90)",
        line=dict(color="#d62728", width=2, dash="dot")
    ))

# Add a vertical line to indicate "Today"
fig.add_vline(x=0, line_width=2, line_dash="solid", line_color="black")
fig.add_annotation(x=0.5, y=0.95, xref="x", yref="paper", text="Forecast Horizon →", showarrow=False)

fig.update_layout(
    xaxis_title="Days Relative to Today",
    yaxis_title="Sales Volume (Units)",
    hovermode="x unified",
    margin=dict(l=20, r=20, t=30, b=20),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    plot_bgcolor="white",
    hoverlabel=dict(bgcolor="white", font_size=13, font_family="Rockwell")
)

# Add faint grid lines
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#e9ecef", zeroline=True, zerolinewidth=2, zerolinecolor="black")
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#e9ecef")

st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# STATISTICAL FOUNDATION FOOTER
# ==============================================================================
st.markdown("---")
with st.expander("ℹ️ Statistical & Quantitative Methodology"):
    st.markdown("""
    **Architecture Breakdown:**
    - **Base Predictions (TFT)**: The initial bounds are generated by a Deep Learning Temporal Fusion Transformer trained with `QuantileLoss([0.1, 0.5, 0.9])`.
    - **Causal Inference (DML)**: The Price Elasticity ($\tau$) is isolated using an Orthogonal/Double Machine Learning Causal Forest. This prevents omitted variable bias (e.g. discounting only during high-demand holidays) from corrupting the true marginal effect of price.
    - **Integration**: The chart reactive formula is $D_{final}[q] = D_{base}[q] \\times \\max(0, 1 + (\\tau \\times \\Delta P))$.
    
    *Notice how sliding the price dynamically scales the volume forecast while preserving the mathematically sound shape of the non-linear uncertainty distribution.*
    """)
