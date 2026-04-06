"""Configuration loading utilities for the Smart Load Shedding Optimizer."""

from __future__ import annotations

import os
from pathlib import Path
import yaml


def default_config() -> dict:
    return {
        "api": {
            "openweathermap": {
                "api_key": "YOUR_API_KEY_HERE",
                "location": "Andhra Pradesh, India",
                "lat": 15.9129,
                "lon": 79.7400,
            }
        },
        "data": {
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "geographic_dir": "data/geographic",
        },
        "data_files": {
            "demand": "data/raw/power_supply_data.csv",
            "outages": "data/raw/outage_data.csv",
            "weather": "data/raw/weather_data.csv",
            "grid_infra": "data/raw/grid_infra_data.csv",
            "critical_infra": "data/raw/crit_infra_data.csv",
        },
        "processed_data_dir": "data/processed",
        "output_dir": "outputs",
        "time_range": {"start_date": "2020-05-14", "end_date": "2023-05-14"},
        "date_range": {"start_date": "2020-05-14", "end_date": "2023-05-14"},
        "forecast": {
            "horizon_days": 7,
            "train_test_split": 0.8,
            "features": [
                "dayofweek",
                "month",
                "is_weekend",
                "temperature_c",
                "is_hot_day",
                "is_rainy_day",
            ],
        },
        "outage_risk": {
            "prediction_window_hours": 24,
            "positive_class_weight": 3,
            "features": [
                "temperature_c",
                "precipitation_mm",
                "utilization_ratio",
                "is_weekend",
                "peak_demand_mw",
            ],
        },
        "load_shedding": {
            "max_percentage_per_district": 30.0,
            "critical_infrastructure_weight": 2.0,
            "max_outage_duration_hours": 4,
            "min_hours_between_outages": 48,
        },
    }


def _normalize_config(config: dict | None) -> dict:
    cfg = dict(config or {})
    base = default_config()

    if "data" not in cfg:
        cfg["data"] = base["data"].copy()
        if "processed_data_dir" in cfg:
            cfg["data"]["processed_dir"] = cfg["processed_data_dir"].rstrip("/")
    else:
        for k, v in base["data"].items():
            cfg["data"].setdefault(k, v)

    cfg.setdefault("data_files", base["data_files"].copy())
    cfg.setdefault("processed_data_dir", cfg["data"]["processed_dir"])
    cfg.setdefault("output_dir", base["output_dir"])

    if "time_range" not in cfg and "date_range" in cfg:
        cfg["time_range"] = cfg["date_range"]
    if "date_range" not in cfg and "time_range" in cfg:
        cfg["date_range"] = cfg["time_range"]
    cfg.setdefault("time_range", base["time_range"].copy())
    cfg.setdefault("date_range", cfg["time_range"])

    for section in ["forecast", "outage_risk", "load_shedding"]:
        merged = base.get(section, {}).copy()
        merged.update(cfg.get(section, {}))
        cfg[section] = merged

    return cfg


def load_config(config_path: str | Path | None = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return default_config()

    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    return _normalize_config(loaded)
