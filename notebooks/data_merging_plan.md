# Data Merging Strategy

## Time Alignment
- Demand data: 15/30/60-minute intervals
- Weather data: hourly measurements
- Outage data: event-based with start/end times
- Infrastructure data: static locations with attributes

## Spatial Alignment
- Match by region_id
- For point data (weather stations, grid assets), assign to containing region

## Feature Engineering Ideas
1. Demand-based:
   - Rolling averages (24h, 7d)
   - Daily peak/off-peak ratio
   - Day-ahead forecast error

2. Weather:
   - Extreme indicators (temp > 35°C, rainfall > 100mm)
   - Heat/cold wave duration

3. Outage-focused:
   - Days since last outage per region
   - Cumulative outage duration last 30 days
   - Binary outage flag (next 24h prediction target)

4. Infrastructure:
   - Asset density per region
   - Average age of equipment
   - Distance to nearest backup supply

5. Grid health:
   - Load factor (avg load / peak load)
   - Reserve margin

## Master Dataset Schema
| Column | Type | Source | Description |
|--------|------|--------|-------------|
| timestamp | datetime | All | Hourly timestamps |
| region_id | string | All | State/district ID |
| demand_mw | float | Demand data | Actual power demand |
| temperature | float | Weather | Temperature in °C |
| ... | ... | ... | ... |