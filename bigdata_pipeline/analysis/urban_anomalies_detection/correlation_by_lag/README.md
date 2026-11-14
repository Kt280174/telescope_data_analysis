# correlation_by_lag

Lag Correlation analysis between Nizhny Novgorod and Novosibirsk.

## How it was generated

`urban_anomalies_detection.py`, run on 2026-11-12.

- Input metadata: `telescope_files/source_channel_metadata.csv`
- Input chains: `telescope_pipeline_files/gold/chains`
- Target Urban Zones: Russian cities (based on keyword matching).
- Anomaly detection: Robust Z-score (threshold > 4.0) on 6-hour intervals.

## Content (7 rows)

| Column | Meaning |
|---|---|
| `City_1` | Primary city |
| `City_2` | Secondary city |
| `Lag_Days` | Time lag in days (negative means City 2 leads) |
| `Correlation` | Pearson correlation coefficient |
