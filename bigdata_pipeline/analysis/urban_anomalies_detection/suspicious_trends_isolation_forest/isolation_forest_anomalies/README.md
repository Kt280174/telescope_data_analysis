# isolation_forest_anomalies

Multivariate anomalies detected via Isolation Forest. High coord_ratio indicates coordinated behavior.

## How it was generated

`isolation_forest_analysis.py`, run on 2026-11-14.

Isolation Forest for coordinated activity detection (Task 5).

## Content (1,040 rows)

| Column | Meaning |
|---|---|
| `date` | Timestamp of the 6-hour window |
| `zone` | Target city |
| `relay_count` | Total relays in the window |
| `unique_channels` | Number of unique forwarding channels |
| `coord_ratio` | Average relays per channel (High = suspicious) |
| `status` | Anomaly classification |
