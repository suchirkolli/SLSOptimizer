# Smart Load Shedding Optimizer: Data Schema (Andhra Pradesh)

## Project Overview

This project focuses on optimizing load shedding in Andhra Pradesh, India, using machine learning to forecast electricity demand, predict outage-prone zones, and recommend optimized load shedding strategies that minimize disruption and improve fairness.

The system uses historical data from May 2020 through May 2023 to develop and demonstrate the optimization approach.

## Geographic Scope

This project analyzes power distribution at the district level across all 13 districts of Andhra Pradesh.

### Geographic Granularity Decision

After evaluating available data sources and considering tradeoffs:

- **District level** (13 districts) has been selected as the geographic granularity
- This provides a good balance between detail and data availability
- State-level data is distributed to districts using population-based weighting

## Common Fields

All processed datasets include these standard fields:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | ISO 8601 format (YYYY-MM-DDTHH:MM:SS) in UTC timezone |
| `district_id` | string | Unique identifier for district (format: AP-XX) |
| `district_name` | string | Human-readable district name |

## Geographic Reference Data

### District Reference (`data/geographic/ap_districts_reference.csv`)

| Field | Type | Description |
|-------|------|-------------|
| `district_id` | string | Unique identifier for each district (format: AP-XX) |
| `district_name` | string | Official name of the district |
| `population` | integer | Total population of district |
| `area_km2` | float | Geographic area in square kilometers |
| `pop_density` | float | Population per square kilometer |
| `pop_proportion` | float | District population as proportion of state total |

## Raw Data Schemas

### Power Supply Data (`data/raw/power_supply_data.csv`)

State-level power statistics from POSOCO and CEA.

| Field | Type | Description |
|-------|------|-------------|
| `Date` | string | Date in DD-MM-YYYY format |
| `Energy Required (MU)` | float | Energy requirement in Million Units |
| `Energy Met (MU)` | float | Energy supplied in Million Units |
| `Energy +/- (MU)` | float | Surplus/deficit in Million Units |
| `Unrestricted Peak Demand (MW)` | float | Peak demand in MegaWatts |
| `Deficit/Surplus (MW)` | float | Peak deficit/surplus in MegaWatts |

### Weather Data (`data/raw/ap_weather.csv`)

State-level weather data from IMD/OpenWeatherMap.

*Note: Weather data is maintained at state level and applied uniformly to all districts*

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Date and time of measurement |
| `temperature` | float | Temperature in degrees Celsius |
| `humidity` | float | Relative humidity percentage |
| `precipitation` | float | Rainfall in mm |
| `wind_speed` | float | Wind speed in m/s |

## Processed Data Schemas

### District Power Data (`data/processed/demand.parquet`)

*Derived from state-level data using population-based distribution*

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Date and time |
| `district_id` | string | District identifier |
| `district_name` | string | District name |
| `demand_mw` | float | Estimated district power demand |
| `supply_mw` | float | Estimated district power supply |
| `deficit_mw` | float | Estimated district power deficit |
| `pop_density` | float | District population density |

### Weather Data (`data/processed/weather.parquet`)

*Applied uniformly to all districts*

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Hourly timestamp |
| `temperature` | float | Temperature in °C |
| `humidity` | float | Relative humidity % |
| `precipitation` | float | Precipitation in mm |
| `wind_speed` | float | Wind speed in m/s |

### Master Dataset (`data/processed/master.parquet`)

The fully merged dataset used for modeling, combining all above sources.

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | datetime | Hourly timestamp |
| `district_id` | string | District identifier |
| `district_name` | string | District name |
| `demand_mw` | float | Electricity demand |
| `supply_mw` | float | Electricity supply |
| `deficit_mw` | float |