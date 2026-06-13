# Prescriptive Demand Forecasting Engine

A production-grade, end-to-end system combining **Temporal Fusion Transformer (TFT)** demand forecasting with **Double Machine Learning (DML)** causal price elasticity estimation, built on the Kaggle M5 Forecasting Dataset (Walmart retail data).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                       │
│         Fan Charts · Pricing Slider · Risk Metrics          │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
    ┌──────────▼──────────┐    ┌──────────▼──────────┐
    │   Predictive Engine │    │    Causal Engine     │
    │   (TFT Quantile)    │    │  (CausalForestDML)  │
    │  D_base ~ q10/50/90 │    │   τ(x) elasticity   │
    └──────────┬──────────┘    └──────────┬──────────┘
               │                          │
               └──────────┬───────────────┘
                          │
              D_final = D_base × (1 + τ·ΔP)
                          │
               ┌──────────▼──────────┐
               │   Data Pipeline     │
               │  Phase 1: Features  │
               └──────────┬──────────┘
                          │
               ┌──────────▼──────────┐
               │   M5 Raw Data       │
               │ sales · calendar ·  │
               │    sell_prices       │
               └─────────────────────┘
```

## Project Phases

| Phase | Module | Status |
|-------|--------|--------|
| 1 | Data Preprocessing & Feature Engineering | ✅ |
| 2 | TFT Predictive Engine | ⬜ |
| 3 | Causal Inference Engine (DML) | ⬜ |
| 4 | Integration & Backtesting | ⬜ |
| 5 | Streamlit Dashboard | ⬜ |

## Dataset

**Kaggle M5 Forecasting – Accuracy**
- `sales_train_validation.csv` — daily unit sales (wide format, ~30K items × 1,913 days)
- `calendar.csv` — date features, events, SNAP indicators
- `sell_prices.csv` — weekly item-store prices

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place M5 data in data/raw/
#    - sales_train_validation.csv
#    - calendar.csv
#    - sell_prices.csv

# 3. Run the data pipeline
python data_pipeline.py --data-dir data/raw --output-dir data/processed
```

## License

MIT
