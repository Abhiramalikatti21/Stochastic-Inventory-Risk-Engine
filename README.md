# Prescriptive Demand Forecasting Engine

An end-to-end, production-grade demand forecasting and causal inference system. Built for the Amazon ML Summer School application and tailored for quantitative finance workflows, this project predicts multi-horizon probabilistic demand and isolates the true marginal effect of price interventions (price elasticity).

This project uses the Kaggle M5 Forecasting Dataset (Walmart retail sales, prices, and calendar events).

## 🌟 Key Features
- **Temporal Fusion Transformers (TFT)**: SOTA Deep Learning for multi-horizon time-series forecasting, utilizing `pytorch-forecasting`. Predicts downside risk (10th percentile), baseline expectation (50th percentile), and upside potential (90th percentile).
- **Double Machine Learning (DML)**: Causal Inference engine using Microsoft's `EconML`. Isolates true price elasticity (τ) by controlling for highly confounded seasonal and promotional variables.
- **Quant Backtesting Evaluation**: Institutional-grade metrics tracking including Weighted Quantile Loss (WQL), Mean Absolute Scaled Error (MASE), and Empirical Probabilistic Coverage.
- **"What-If" Simulator Dashboard**: An interactive Streamlit and Plotly web app that fuses probabilistic deep learning with causal trees to instantly simulate the impact of pricing strategies on expected revenue and stockout risk.

---

## 🛠️ System Architecture & Pipeline

The project is structured into 5 modular, production-ready phases:

### Phase 1: Data Preprocessing & Feature Engineering
- **File**: `data_pipeline.py`
- Generates temporal rolling statistics, historical lags, price momentum indicators, and encoded calendar events. 
- Handles the Kaggle M5 hierarchical format to output a fully-featured Parquet file.

### Phase 2: The Predictive Engine (Deep Learning)
- **File**: `predictive_engine.py`
- Maps Phase 1 features to a TFT topology (separating known vs. unknown future variables).
- Fits a PyTorch Lightning model optimizing for `QuantileLoss([0.1, 0.5, 0.9])`.

### Phase 3: The Causal Inference Engine
- **File**: `causal_engine.py`
- Implements a `CausalForestDML` architecture.
- Uses `LightGBMRegressor` nuisance models to orthogonalize the outcome (Sales) and treatment (Price).
- Calculates the Conditional Average Treatment Effect (CATE) to uncover exactly *why* and *how much* products react to price changes.

### Phase 4: Integration & Evaluation
- **File**: `integration_eval.py`
- Fuses Phase 2 base demand predictions with Phase 3 elasticities: D_final = D_base * max(0, 1 + (τ * ΔP))
- Generates a `evaluation_report.json` detailing WQL, MASE, and Coverage.

### Phase 5: The Streamlit Dashboard
- **File**: `app.py`
- Interactive frontend allowing business stakeholders and quant researchers to slide price interventions (ΔP) and instantly visualize the shifting demand curve in a probabilistic fan chart.

---

## 🚀 Quick Start Guide

### 1. Installation
Clone this repository and install the deep learning and causal inference dependencies.
```bash
pip install -r requirements.txt
2. Run the Interactive Dashboard
The fastest way to see the project in action is through the Streamlit dashboard, which automatically loads mock/cached data for a seamless demo:

text


python -m streamlit run app.py
3. Run the Full Backend Pipeline
To execute the pipeline end-to-end on your local machine:

text


# 1. Process data & engineer features
python data_pipeline.py --data-dir data/raw --output-dir data/processed
# 2. Train the Predictive Engine (TFT)
python predictive_engine.py --data-path data/processed/full_featured.parquet
# 3. Evaluate the system integration
python test_integration.py
🧪 Testing
The codebase uses synthetic smoke-tests to ensure structural integrity across all components without requiring hours of GPU training:

text


python test_pipeline.py
python test_predictive_engine.py
python test_causal_engine.py
python test_integration.py
