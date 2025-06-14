# SLSOptimizer
Smart Load-Shedding Optimizer POC

An end-to-end, two-month proof-of-concept that ingests Indian power-grid data, builds regional demand forecasts & outage-risk classifiers, then generates a fair, rotational load-shedding schedule—all wrapped in a lightweight dashboard.

---

## 🚀 Features

- **Data Ingestion & ETL**  
  Pulls historical demand, outage logs, weather (OpenWeatherMap/IMD), calendar events & social feeds (Twitter).

- **Demand Forecasting**  
  Compares ARIMA, XGBoost (or LSTM) to predict short-term, region-level electricity demand.

- **Outage-Risk Classification**  
  Trains Random Forest / LightGBM models to score zones by likelihood of equipment failures or overloads.

- **Load-Shedding Optimizer**  
  Rule-based (with optional RL sketch) engine that:  
  1. Meets a target supply cut  
  2. Minimizes social/economic cost  
  3. Ensures rotation fairness & critical-service prioritization

- **Interactive Dashboard**  
  Streamlit/Flask app showing:  
  - Demand-forecast time series  
  - Geographic risk heatmap  
  - Next-day shedding schedule
