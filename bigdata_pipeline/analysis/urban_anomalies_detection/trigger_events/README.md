# trigger_events

List of Trigger Events segmented from anomalous activity peaks.

## How it was generated

`urban_anomalies_detection.py`, run on 2026-11-12.

- Input metadata: `telescope_files/source_channel_metadata.csv`
- Input chains: `telescope_pipeline_files/gold/chains`
- Target Urban Zones: Russian cities (based on keyword matching).
- Anomaly detection: Robust Z-score (threshold > 4.0) on 6-hour intervals.

## Content (14,672 rows)

| Column | Meaning |
|---|---|
| `zone` | Target city name |
| `start_time` | Start time of the event |
| `end_time` | End time of the event |
| `duration_6h_blocks` | Duration in 6-hour blocks |
| `peak_relays` | Highest number of relays at the peak |
| `max_z_score` | Maximum Z-score (severity of the anomaly) |
