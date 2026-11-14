# correlation_matrix

Pearson correlation matrix of daily forwarding activity between the top 5 cities.

## How it was generated

`urban_anomalies_detection.py`, run on 2026-11-12.

- Input metadata: `telescope_files/source_channel_metadata.csv`
- Input chains: `telescope_pipeline_files/gold/chains`
- Target Urban Zones: Russian cities (based on keyword matching).
- Anomaly detection: Robust Z-score (threshold > 4.0) on 6-hour intervals.

## Content (5 rows)

| Column | Meaning |
|---|---|
| `Nizhny Novgorod` | Correlation with Nizhny Novgorod |
| `Novosibirsk` | Correlation with Novosibirsk |
| `Moscow` | Correlation with Moscow |
| `Khabarovsk` | Correlation with Khabarovsk |
| `Saint Petersburg` | Correlation with Saint Petersburg |
