import pandas as pd
import numpy as np
import glob
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import date
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

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
        
        cols = "\n".join(f"| `{c}` | {t} |" for c, t in columns.items())
        readme_text = (
            f"# {name}\n\n{what}\n\n## How it was generated\n\n"
            f"`isolation_forest_analysis.py`, run on {date.today().isoformat()}.\n\n{self.context}\n\n"
            f"## Content ({len(df):,} rows)\n\n| Column | Meaning |\n|---|---|\n{cols}\n"
        )
        (d / "README.md").write_text(readme_text, encoding="utf-8")
        self.tables.append(name)
        print(f"  wrote {name}/ ({len(df):,} rows)")

# ==========================================
# 1. READ METADATA AND ASSIGN CITIES
# ==========================================
print("1. Reading Metadata...")
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
# 3. FEATURE ENGINEERING & ISOLATION FOREST
# ==========================================
out_dir = "bigdata_pipeline/analysis/urban_anomalies_detection/suspicious_trends_isolation_forest"
W = Writer(out_dir, "Isolation Forest for coordinated activity detection (Task 5).")
fig_dir = Path(out_dir) / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

print("3. Extracting features for Isolation Forest...")
features_list = []

for zone in df_merged['zone'].unique():
    df_zone = df_merged[df_merged['zone'] == zone]
    
    ts = df_zone.set_index('date').resample('6h').agg(
        relay_count=('dest_channel', 'count'),
        unique_channels=('dest_channel', 'nunique')
    )
    
    # CHỈ GIỮ LẠI CÁC KHUNG GIỜ CÓ HOẠT ĐỘNG (>0) để tránh nhiễu
    ts = ts[ts['relay_count'] > 0]
    
    # Tính toán đặc trưng
    ts['coord_ratio'] = ts['relay_count'] / ts['unique_channels']
    ts['zone'] = zone
    features_list.append(ts.reset_index())

df_features = pd.concat(features_list).dropna()

# SỬA LỖI INDEX TRÙNG LẶP: Reset index để tạo dãy số nguyên duy nhất
df_features = df_features.reset_index(drop=True)

# Lọc bỏ các ngoại lai cực đoan (top 1% relay) để mô hình bớt bị ảnh hưởng
q = df_features['relay_count'].quantile(0.99)
df_filtered = df_features[df_features['relay_count'] <= q].copy()

# Chuẩn bị dữ liệu cho mô hình
X = df_filtered[['relay_count', 'unique_channels', 'coord_ratio']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"4. Training Isolation Forest on {len(df_filtered)} active windows...")
# Giảm contamination xuống 1% để chỉ bắt các đỉnh thực sự chót vót
iso_forest = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
df_filtered['anomaly'] = iso_forest.fit_predict(X_scaled)

# Gán nhãn
df_filtered['status'] = df_filtered['anomaly'].map({-1: 'Suspicious', 1: 'Normal'})

# Cập nhật nhãn vào bảng lớn (an toàn vì index đã được reset)
df_features['status'] = 'Normal'
df_features.loc[df_filtered.index, 'status'] = df_filtered['status']

# Lấy các sự kiện đáng ngờ
df_suspicious_if = df_features[df_features['status'] == 'Suspicious'].copy()
df_suspicious_if = df_suspicious_if.sort_values('coord_ratio', ascending=False)

# Lưu file CSV
W.table("isolation_forest_anomalies", df_suspicious_if,
        "Multivariate anomalies detected via Isolation Forest. High coord_ratio indicates coordinated behavior.",
        {"date": "Timestamp of the 6-hour window", "zone": "Target city", 
         "relay_count": "Total relays in the window", "unique_channels": "Number of unique forwarding channels",
         "coord_ratio": "Average relays per channel (High = suspicious)", "status": "Anomaly classification"})

# ==========================================
# 4. PLOTTING (RUSSIAN LABELS ONLY)
# ==========================================
print("5. Plotting charts...")

# Chart 1: Scatter Plot
plt.figure(figsize=(10, 6))
normal = df_features[df_features['status'] == 'Normal']
suspicious = df_features[df_features['status'] == 'Suspicious']

plt.scatter(normal['relay_count'], normal['unique_channels'], 
            color='#1f77b4', alpha=0.3, label='Нормальная активность')
plt.scatter(suspicious['relay_count'], suspicious['unique_channels'], 
            color='red', edgecolors='black', alpha=0.9, label='Подозрительная координация', s=50)

plt.xlabel('Количество пересылок (Relay Count)', fontsize=12)
plt.ylabel('Уникальные каналы (Unique Channels)', fontsize=12)
plt.title('Выявление скоординированной активности (Isolation Forest)', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(fig_dir / "1_isolation_forest_scatter.png", dpi=200)
plt.close()

# Chart 2: Top 10 Coordinated Events
plt.figure(figsize=(12, 6))
top_coord = df_suspicious_if.head(10).copy()
if len(top_coord) > 0:
    top_coord['label'] = top_coord['zone'] + " (" + top_coord['date'].dt.strftime('%Y-%m-%d %H:%M') + ")"
    bars = plt.barh(top_coord['label'], top_coord['coord_ratio'], color='darkorange')
    for bar, channels in zip(bars, top_coord['unique_channels']):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                 f"Каналов: {channels}", va='center', fontsize=9)
    plt.xlabel('Коэффициент координации (Relays / Channels)', fontsize=12)
    plt.title('Топ-10 скоординированных событий (Высокий уровень координации)', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(fig_dir / "2_top_coordinated_events.png", dpi=200)
plt.close()

# README cho figures
fig_readme = """# Figures - Isolation Forest (Suspicious Trends)

| File | What it shows |
|---|---|
| `1_isolation_forest_scatter.png` | Scatter plot of Relay Count vs Unique Channels. Red dots are isolated as suspicious. |
| `2_top_coordinated_events.png` | Top 10 events with the highest coordination ratio. |
"""
(fig_dir / "README.md").write_text(fig_readme, encoding="utf-8")

print(f"\n[SUCCESS] All results saved to: {out_dir}")