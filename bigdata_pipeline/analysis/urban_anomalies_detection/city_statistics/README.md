# city_statistics

Overview of forwarding activity for each target Russian city.

## How it was generated

`urban_anomalies_detection.py`, run on 2026-11-12.

- Input metadata: `telescope_files/source_channel_metadata.csv`
- Input chains: `telescope_pipeline_files/gold/chains`
- Target Urban Zones: Russian cities (based on keyword matching).
- Anomaly detection: Robust Z-score (threshold > 4.0) on 6-hour intervals.

## Content (24 rows)

| Column | Meaning |
|---|---|
| `zone` | Target city name |
| `total_relays` | Total number of forwarded relays |
| `start_date` | Start date of collected data |
| `end_date` | End date of collected data |
| `unique_channels` | Number of unique active channels in the city |
