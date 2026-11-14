import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
from datetime import date
from pathlib import Path

# ==========================================
# 0. WRITER CLASS (PROFESSIONAL STORAGE)
# ==========================================
class Writer:
    def __init__(self, out, context):
        self.out, self.context = Path(out), context
        self.tables = []
        self.out.mkdir(parents=True, exist_ok=True)

    def table(self, name, df, what, columns):
        d = self.out / name
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / f"{name}.csv", index=False)
        
        # Generate README.md content
        cols = "\n".join(f"| `{c}` | {t} |" for c, t in columns.items())
        readme_text = (
            f"# {name}\n\n{what}\n\n## How it was generated\n\n"
            f"`urban_anomalies_detection.py`, run on {date.today().isoformat()}.\n\n{self.context}\n\n"
            f"## Content ({len(df):,} rows)\n\n| Column | Meaning |\n|---|---|\n{cols}\n"
        )
        (d / "README.md").write_text(readme_text, encoding="utf-8")
        self.tables.append(name)
        print(f"  wrote {name}/ ({len(df):,} rows)")

# ==========================================
# 1. READ METADATA AND ASSIGN CITIES
# ==========================================
print("1. Reading Metadata (Parsing JSON)...")
meta_path = "telescope_files/source_channel_metadata.csv" 

if not os.path.exists(meta_path):
    print(f"[ERROR] File not found at {meta_path}.")
    exit()

df_meta = pd.read_csv(meta_path, dtype={'id': str})

def parse_json_data(json_str):
    try: return json.loads(json_str)
    except: return {}

df_meta['parsed'] = df_meta['data'].apply(parse_json_data)
df_meta['channel_name'] = df_meta['parsed'].apply(lambda d: d.get('title') or d.get('username'))
df_meta['channel_id'] = pd.to_numeric(df_meta['id'].str.extract(r'(\d+)')[0], errors='coerce')
df_meta = df_meta.dropna(subset=['channel_id', 'channel_name'])
df_meta['channel_id'] = df_meta['channel_id'].astype('int64')

city_keywords = {
    'Moscow': ['moscow', 'moskva', 'москва', 'мск'],
    'Saint Petersburg': ['saint petersburg', 'petersburg', 'spb', 'санкт-петербург', 'петербург', 'спб', 'питер'],
    'Novosibirsk': ['novosibirsk', 'новосибирск', 'нск'],
    'Yekaterinburg': ['yekaterinburg', 'ekaterinburg', 'екатеринбург', 'екб'],
    'Kazan': ['kazan', 'казань'], 'Nizhny Novgorod': ['nizhny novgorod', 'нижний новгород', 'нн'],
    'Chelyabinsk': ['chelyabinsk', 'челябинск'], 'Omsk': ['omsk', 'омск'], 'Samara': ['samara', 'самара'],
    'Rostov-on-Don': ['rostov-on-don', 'rostov', 'ростов-на-дону', 'ростов'], 'Ufa': ['ufa', 'уфа'],
    'Krasnoyarsk': ['krasnoyarsk', 'красноярск'], 'Voronezh': ['voronezh', 'воронеж'], 'Perm': ['perm', 'пермь'],
    'Volgograd': ['volgograd', 'волгоград'], 'Krasnodar': ['krasnodar', 'краснодар'], 'Saratov': ['saratov', 'саратов'],
    'Tyumen': ['tyumen', 'тюмень'], 'Irkutsk': ['irkutsk', 'иркутск'], 'Khabarovsk': ['khabarovsk', 'хабаровск'],
    'Vladivostok': ['vladivostok', 'владивосток'], 'Sochi': ['sochi', 'сочи'], 'Kaliningrad': ['kaliningrad', 'калининград'],
    'Belgorod': ['belgorod', 'белгород'], 'Murmansk': ['murmansk', 'мурманск'], 'Magadan': ['magadan', 'магадан']
}

def assign_zone(title):
    if pd.isna(title): return None
    title_lower = str(title).lower()
    for zone, keywords in city_keywords.items():
        for kw in keywords:
            if kw in title_lower: return zone
    return None

df_meta['zone'] = df_meta['channel_name'].apply(assign_zone)
df_meta_zone = df_meta.dropna(subset=['zone'])[['channel_id', 'zone']]

# ==========================================
# 2. READ PARQUET AND MERGE DATA
# ==========================================
print("2. Reading Parquet and merging data...")
files = glob.glob("telescope_pipeline_files/gold/chains/**/*.parquet", recursive=True)
df_chains = pd.concat([pd.read_parquet(f, columns=['date', 'dest_channel']) for f in files])
df_chains['dest_channel'] = df_chains['dest_channel'].astype('int64')

df_merged = df_chains.merge(df_meta_zone, left_on='dest_channel', right_on='channel_id', how='inner')
df_merged['date'] = pd.to_datetime(df_merged['date'], unit='s', errors='coerce')
df_merged = df_merged.dropna(subset=['date'])

# ==========================================
# 3. INITIALIZE WRITER AND ANALYZE
# ==========================================
out_dir = "bigdata_pipeline/analysis/urban_anomalies_detection"
fig_dir = Path(out_dir) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

context = (
    f"- Input metadata: `{meta_path}`\n"
    f"- Input chains: `telescope_pipeline_files/gold/chains`\n"
    f"- Target Urban Zones: Russian cities (based on keyword matching).\n"
    f"- Anomaly detection: Robust Z-score (threshold > 4.0) on 6-hour intervals."
)

W = Writer(out_dir, context)

# ------------------------------------------
# TABLE 1: CITY STATISTICS
# ------------------------------------------
print("3. Calculating city statistics...")
city_stats = df_merged.groupby('zone').agg(
    total_relays=('date', 'count'),
    start_date=('date', 'min'),
    end_date=('date', 'max'),
    unique_channels=('dest_channel', 'nunique')
).reset_index().sort_values('total_relays', ascending=False)

W.table("city_statistics", city_stats,
        "Overview of forwarding activity for each target Russian city.",
        {"zone": "Target city name", "total_relays": "Total number of forwarded relays", 
         "start_date": "Start date of collected data", "end_date": "End date of collected data", 
         "unique_channels": "Number of unique active channels in the city"})

# ------------------------------------------
# TABLE 2: TRIGGER EVENTS
# ------------------------------------------
print("4. Detecting temporal anomalies and segmenting events...")
all_segments = []
for zone in df_merged['zone'].unique():
    df_zone = df_merged[df_merged['zone'] == zone].set_index('date')
    ts = df_zone.resample('6h').size().reset_index(name='count')
    
    if len(ts) < 10: continue
    
    median = ts['count'].median()
    mad = np.median(np.abs(ts['count'] - median)) or 1
    ts['z_score'] = 0.6745 * (ts['count'] - median) / mad
    ts['is_anomaly'] = ts['z_score'] > 4.0 
    
    ts['anomaly_group'] = ts['is_anomaly'].ne(ts['is_anomaly'].shift()).cumsum()
    segments = ts[ts['is_anomaly'] == True].groupby('anomaly_group').agg(
        zone=('date', lambda x: zone),
        start_time=('date', 'min'),
        end_time=('date', 'max'),
        duration_6h_blocks=('date', 'count'),
        peak_relays=('count', 'max'),
        max_z_score=('z_score', 'max')
    ).reset_index(drop=True)
    all_segments.append(segments)

# Gộp tất cả sự kiện
df_events = pd.concat(all_segments).sort_values('max_z_score', ascending=False)

# Lưu toàn bộ sự kiện (Bảng gốc)
W.table("trigger_events", df_events,
        "List of Trigger Events segmented from anomalous activity peaks.",
        {"zone": "Target city name", "start_time": "Start time of the event", "end_time": "End time of the event", 
         "duration_6h_blocks": "Duration in 6-hour blocks", "peak_relays": "Highest number of relays at the peak", 
         "max_z_score": "Maximum Z-score (severity of the anomaly)"})

# ---> TẠO BẢNG MỚI: TOP 10 SỰ KIỆN BÙNG NỔ NGẮN HẠN (Lọc < 20 blocks ~ 5 ngày)
df_top_10 = df_events[df_events['duration_6h_blocks'] <= 20].head(10).copy()

W.table("top_10_trigger_events", df_top_10,
        "Top 10 most intense, short-lived trigger events (filtered to max 5 days duration).",
        {"zone": "Target city name", "start_time": "Start time of the event", "end_time": "End time of the event", 
         "duration_6h_blocks": "Duration in 6-hour blocks", "peak_relays": "Highest number of relays at the peak", 
         "max_z_score": "Maximum Z-score (severity of the anomaly)"})

# ------------------------------------------
# TABLE 3 & 4: CORRELATION
# ------------------------------------------
print("5. Calculating correlations between cities...")
top_5_cities = city_stats.head(5)['zone'].tolist()
df_corr_daily = pd.DataFrame()

for city in top_5_cities:
    df_corr_daily[city] = df_merged[df_merged['zone'] == city].set_index('date').resample('1D').size()
df_corr_daily = df_corr_daily.fillna(0)

corr_matrix = df_corr_daily.corr()
W.table("correlation_matrix", corr_matrix.reset_index().rename(columns={'index': 'City'}),
        "Pearson correlation matrix of daily forwarding activity between the top 5 cities.",
        {col: f"Correlation with {col}" for col in corr_matrix.columns})

lag_results = []
city1, city2 = top_5_cities[0], top_5_cities[1]
for lag in range(-3, 4):
    corr = df_corr_daily[city1].corr(df_corr_daily[city2].shift(lag))
    lag_results.append({'City_1': city1, 'City_2': city2, 'Lag_Days': lag, 'Correlation': round(corr, 4)})

df_lag = pd.DataFrame(lag_results)
W.table("correlation_by_lag", df_lag,
        f"Lag Correlation analysis between {city1} and {city2}.",
        {"City_1": "Primary city", "City_2": "Secondary city", 
         "Lag_Days": "Time lag in days (negative means City 2 leads)", "Correlation": "Pearson correlation coefficient"})

# ==========================================
# 6. PLOTTING (RUSSIAN TITLES)
# ==========================================
print("6. Plotting charts...")

# Chart 1: Time series of Top 5
plt.figure(figsize=(15, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for i, zone in enumerate(top_5_cities):
    df_zone = df_merged[df_merged['zone'] == zone].set_index('date')
    ts = df_zone.resample('6h').size().reset_index(name='count')
    median = ts['count'].median()
    mad = np.median(np.abs(ts['count'] - median)) or 1
    ts['z_score'] = 0.6745 * (ts['count'] - median) / mad
    anomalies = ts[ts['z_score'] > 4.0]
    
    plt.plot(ts['date'], ts['count'], label=zone, alpha=0.8, color=colors[i])
    plt.scatter(anomalies['date'], anomalies['count'], color='black', s=15)

plt.title('Выявление временных аномалий в целевых городских зонах РФ', fontsize=14)
plt.xlabel('Дата', fontsize=12)
plt.ylabel('Количество пересылок (за 6 часов)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(fig_dir / "1_timeseries_anomalies.png", dpi=200)
plt.close()

# Chart 2: Number of events per city
plt.figure(figsize=(10, 5))
event_counts = df_events['zone'].value_counts().head(10)
plt.bar(event_counts.index, event_counts.values, color='teal')
plt.title('Количество триггерных событий по городам (Сегментация)', fontsize=14)
plt.xlabel('Город', fontsize=12)
plt.ylabel('Количество событий', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(fig_dir / "2_event_distribution.png", dpi=200)
plt.close()

# Chart 3: Correlation Heatmap
plt.figure(figsize=(8, 6))
im = plt.imshow(corr_matrix.values, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(im, label='Коэффициент корреляции')
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45, ha='right')
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
for i in range(len(corr_matrix)):
    for j in range(len(corr_matrix)):
        plt.text(j, i, f"{corr_matrix.values[i, j]:.2f}", ha='center', va='center', color='black')
plt.title('Матрица корреляции активности между городами', fontsize=14)
plt.tight_layout()
plt.savefig(fig_dir / "3_correlation_heatmap.png", dpi=200)
plt.close()

# Chart 4: Top 10 Trigger Events
plt.figure(figsize=(12, 6))
# Tạo nhãn "Thành phố (Ngày)"
labels_top10 = df_top_10['zone'] + " (" + df_top_10['start_time'].dt.strftime('%Y-%m-%d') + ")"
bars = plt.barh(labels_top10, df_top_10['max_z_score'], color='indianred')

# Thêm text thông số (Z-score và Peak Relays) vào biểu đồ
for bar, peak in zip(bars, df_top_10['peak_relays']):
    plt.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2, 
             f"Z={bar.get_width():.0f} (Пик: {peak:,})", 
             va='center', fontsize=9)

plt.xlabel('Уровень Z-score (Аномальность)', fontsize=12)
plt.title('Топ-10 ключевых триггерных событий в РФ (краткосрочные пики)', fontsize=14)
plt.gca().invert_yaxis() # Đảo ngược để sự kiện lớn nhất nằm trên cùng
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(fig_dir / "4_top_10_events.png", dpi=200)
plt.close()


# Generate README for figures folder
fig_readme = """# Figures - Temporal Anomaly & Correlation

Images only. Tables and their detailed READMEs are in the parent folder.

| File | What it shows |
|---|---|
| `1_timeseries_anomalies.png` | Time series of relay activity with black dots indicating detected anomalies (Z > 4.0). |
| `2_event_distribution.png` | Bar chart showing how many trigger events were detected per city. |
| `3_correlation_heatmap.png` | Heatmap showing Pearson correlation of daily activity between the top 5 cities. |
| `4_top_10_events.png` | Bar chart of the Top 10 most intense trigger events (filtered to max 5 days duration). |
"""
(fig_dir / "README.md").write_text(fig_readme, encoding="utf-8")

print(f"\n[SUCCESS] All results saved to: {out_dir}")
print("Each table has its own README.md explaining the columns.")