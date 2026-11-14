import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import date

# Đường dẫn đến thư mục kết quả gốc
out_dir = "bigdata_pipeline/analysis/urban_anomalies_detection"
suspicious_dir = Path(out_dir) / "suspicious_trends"
suspicious_dir.mkdir(parents=True, exist_ok=True) # Tự động tạo folder

# ==========================================
# 1. LOAD DATA
# ==========================================
print("Loading data...")
df_events = pd.read_csv(f"{out_dir}/trigger_events/trigger_events.csv")
df_stats = pd.read_csv(f"{out_dir}/city_statistics/city_statistics.csv")

# Convert time and extract month
df_events['start_time'] = pd.to_datetime(df_events['start_time'])
df_events['month'] = df_events['start_time'].dt.month

# ==========================================
# 2. ANALYSIS: EXTREME Z-SCORES (> 300)
# ==========================================
print("\n--- 1. Extremely high Z-score values (> 300) ---")
extreme_z = df_events[df_events['max_z_score'] > 300].sort_values('max_z_score', ascending=False)
print(extreme_z[['zone', 'start_time', 'peak_relays', 'max_z_score']].head(5).to_string(index=False))

# ==========================================
# 3. ANALYSIS: SMALL NUMBER EFFECT
# ==========================================
print("\n--- 2. Small Number Effect (Low activity cities) ---")
df_merged_events = df_events.merge(df_stats[['zone', 'total_relays']], on='zone', how='left')

# Filter cities with total relays < 50,000 but high Z-score
small_numbers = df_merged_events[(df_merged_events['total_relays'] < 50000) & 
                                 (df_merged_events['max_z_score'] > 10)].sort_values('max_z_score', ascending=False)
print(small_numbers[['zone', 'start_time', 'total_relays', 'peak_relays', 'max_z_score']].head(5).to_string(index=False))

# ==========================================
# 4. ANALYSIS: SEASONAL SPIKES (Summer Months)
# ==========================================
print("\n--- 3. Seasonal Spikes (June, July, August) ---")
summer_events = df_events[df_events['month'].isin([6, 7, 8])].sort_values('max_z_score', ascending=False)
summer_high = summer_events[summer_events['max_z_score'] > 20]
print(summer_high[['zone', 'start_time', 'peak_relays', 'max_z_score']].head(5).to_string(index=False))

# ==========================================
# 5. ESTIMATE FALSE POSITIVE RATE
# ==========================================
print("\n--- 4. Estimating False Positive Rate ---")
total_events = len(df_events)
suspicious_z = len(extreme_z)
suspicious_small = len(small_numbers)

df_suspicious = pd.concat([extreme_z, small_numbers]).drop_duplicates()
num_suspicious = len(df_suspicious)

fpr_estimate = (num_suspicious / total_events) * 100

print(f"Total events: {total_events:,}")
print(f"Suspicious events (Z > 300 or Small Cities): {num_suspicious:,}")
print(f"=> Estimated False Positive Rate: {fpr_estimate:.1f}%")

# Save to CSV
output_file = suspicious_dir / "suspicious_trends.csv"
df_suspicious.to_csv(output_file, index=False)
print(f"\n=> Saved suspicious events to: {output_file}")

# ==========================================
# 6. PLOTTING (RUSSIAN LABELS ONLY)
# ==========================================
print("\nPlotting charts...")
fig_dir_suspicious = suspicious_dir / "figures"
fig_dir_suspicious.mkdir(parents=True, exist_ok=True)

# Chart 1: Scatter Plot (Peak Relays vs Z-score)
plt.figure(figsize=(10, 6))

# Plot normal events (Blue)
normal_events = df_merged_events[~df_merged_events.index.isin(df_suspicious.index)]
plt.scatter(normal_events['peak_relays'], normal_events['max_z_score'], 
            color='#1f77b4', alpha=0.5, label='Обычные события')

# Plot suspicious events (Red)
plt.scatter(df_suspicious['peak_relays'], df_suspicious['max_z_score'], 
            color='red', edgecolors='black', alpha=0.8, label='Подозрительные (ложные)', s=60)

plt.xlabel('Пик пересылок (Peak Relays)', fontsize=12)
plt.ylabel('Уровень Z-score', fontsize=12)
plt.title('Анализ аномалий: Истинные события vs Ложные срабатывания', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(fig_dir_suspicious / "1_suspicious_scatter.png", dpi=200)
plt.close()

# Chart 2: Pie Chart (False Positive Rate)
plt.figure(figsize=(7, 7))
labels = ['Истинные события', 'Подозрительные события']
sizes = [total_events - num_suspicious, num_suspicious]
colors = ['#86B3A1', '#C0504D']
explode = (0, 0.1) 

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
        shadow=True, startangle=140, textprops={'fontsize': 12})
plt.title('Оценка доли ложных срабатываний', fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir_suspicious / "2_false_positive_pie.png", dpi=200)
plt.close()

# Update README for figures folder
fig_readme = """# Figures - Suspicious Trends Analysis

Visualizations for the analysis of potential false positives and suspicious patterns.

| File | What it shows |
|---|---|
| `1_suspicious_scatter.png` | Scatter plot of Peak Relays vs Z-score, highlighting potential false positives (red dots). |
| `2_false_positive_pie.png` | Pie chart showing the estimated proportion of false positives among all detected events. |
"""
(fig_dir_suspicious / "README.md").write_text(fig_readme, encoding="utf-8")

# Update README for suspicious_trends folder
readme_text = f"""# suspicious_trends

Analysis of suspicious patterns and potential false positives in the detected trigger events.

## How it was generated

`suspicious_trends_analysis.py`, run on {date.today().isoformat()}.

## Content ({len(df_suspicious):,} rows)

This table contains events classified as potentially suspicious due to:
- Extremely high Z-scores (baseline shift)
- Small number effect (low activity cities)
- Possible seasonal variations
"""

(suspicious_dir / "README.md").write_text(readme_text, encoding="utf-8")

print(f"=> Charts saved to: {fig_dir_suspicious}")
print(f"=> READMEs generated.")