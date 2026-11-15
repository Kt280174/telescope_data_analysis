#!/usr/bin/env python3
"""
Theme 4 - maps of the activity anomalies and of the links between cities.

Reads the tables written by anomaly_peaks.py and draws four interactive maps (HTML, plotly):

  1. map_anomaly_activity.html     where the anomalies are: one bubble per city, size = number of episodes,
                                   colour = excess posts (log scale); hover gives the split spike / short / sustained.
  2. map_city_correlation.html     which cities move together: lines between the most correlated pairs
                                   (width = correlation, orange = one city follows the other by 1-3 days,
                                   blue = same day; a drop-down filters the links); hover on the middle dot gives the pair and who comes first.
  3. map_city_links.html           the same links one city at a time: a drop-down picks a city and the map shows
                                   its most correlated partners like a star (easier to read than map 2).
  4. map_synchronized_peaks.html   how a synchronized peak spreads: for each of the largest peaks (drop-down),
                                   the cities in the peak coloured by the hours after the first city; the first city
                                   is a star. Needs sync_onsets (run job5 with --hourly, then anomaly_peaks.py).

City positions come from city_coordinates.csv (country, city, lat, lon); cities without a position are listed
and skipped.  Requires: pandas, numpy, plotly  (pip install plotly).  The HTML files load plotly.js from a CDN
(small files, internet needed to open them); use --offline to embed it (about 4 MB per file).

Usage:
  python3 bigdata_pipeline/analysis/anomaly_maps.py \
    --anomalies bigdata_pipeline/analysis/anomalies \
    --coords bigdata_pipeline/analysis/city_coordinates/city_coordinates.csv \
    --figdir bigdata_pipeline/analysis/figures/anomalies
"""
from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KEY = re.compile(r"^(.*) \((.*)\)$")


# ------------------------------------------------------------------------------------------ data
def load_table(folder, name: str):
    p = Path(folder) / name / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def load_coords(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["city_key"] = df["city"].astype(str) + " (" + df["country"].astype(str) + ")"
    return df.drop_duplicates("city_key").set_index("city_key")[["lat", "lon"]]


def with_coords(df: pd.DataFrame, coords: pd.DataFrame, col: str = "city_key"):
    """Add lat/lon; returns (frame with positions, sorted list of cities without a position)."""
    out = df.join(coords, on=col)
    missing = sorted(out.loc[out.lat.isna(), col].unique())
    return out.dropna(subset=["lat", "lon"]).copy(), missing


def city_summary(eps: pd.DataFrame) -> pd.DataFrame:
    """One row per city: number of episodes, anomalous days, excess posts, split by shape."""
    g = eps.groupby("city_key")
    s = pd.DataFrame({
        "n_episodes": g.size(), "n_days": g.n_days.sum(), "excess_posts": g.excess_posts.sum(),
        "max_ratio": g.peak_ratio.max(),
    })
    shapes = eps.pivot_table(index="city_key", columns="shape", values="n_days", aggfunc="size", fill_value=0)
    for c in ("spike", "short", "sustained"):
        s[c] = shapes[c] if c in shapes else 0
    return s.reset_index()


def strongest_pairs(corr: pd.DataFrame, coords: pd.DataFrame, min_corr: float = 0.35, top: int = 80,
                    per_city: int = 4) -> pd.DataFrame:
    """Most correlated city pairs with positions and the leading city ('' when same day).
    A city keeps at most `per_city` links so that a few busy cities do not hide the others."""
    c = corr[corr.best_corr >= min_corr].sort_values("best_corr", ascending=False)
    c = c.join(coords.add_suffix("_a"), on="city_a").join(coords.add_suffix("_b"), on="city_b")
    c = c.dropna(subset=["lat_a", "lon_a", "lat_b", "lon_b"])
    used, keep = {}, []
    for r in c.itertuples():
        if used.get(r.city_a, 0) < per_city and used.get(r.city_b, 0) < per_city:
            keep.append(r.Index)
            used[r.city_a] = used.get(r.city_a, 0) + 1
            used[r.city_b] = used.get(r.city_b, 0) + 1
            if len(keep) >= top:
                break
    c = c.loc[keep].copy()
    # lag > 0: city_b reacts best_lag_days days after city_a
    c["leader"] = np.where(c.best_lag_days > 0, c.city_a, np.where(c.best_lag_days < 0, c.city_b, ""))
    return c.reset_index(drop=True)


def country_centroids(coords: pd.DataFrame, countries) -> pd.DataFrame:
    """Position of a country label: mean position of the known cities of that country."""
    t = coords.copy()
    t["country"] = [KEY.match(k).group(2) if KEY.match(k) else "" for k in t.index]
    t = t[t.country.isin(set(countries))]
    return t.groupby("country")[["lat", "lon"]].mean().reset_index()


def peak_ranking(sync: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    return sync.sort_values(["n_cities", "excess_posts"], ascending=False).head(top)


# ------------------------------------------------------------------------------------------ figures
def view_range(df: pd.DataFrame) -> dict:
    """Latitude / longitude window that fits the cities of `df` with a margin."""
    la, lo = df.lat.agg(["min", "max"]), df.lon.agg(["min", "max"])
    pl, po = max(3.0, 0.12 * (la["max"] - la["min"])), max(5.0, 0.08 * (lo["max"] - lo["min"]))
    return {"geo.lataxis.range": [float(max(-85, la["min"] - pl)), float(min(85, la["max"] + pl))],
            "geo.lonaxis.range": [float(max(-180, lo["min"] - po)), float(min(180, lo["max"] + po))]}


def _geo(fig, title: str, view: pd.DataFrame) -> None:
    r = view_range(view)
    fig.update_geos(projection_type="natural earth", showcountries=True, countrycolor="#a8a8a8",
                    showland=True, landcolor="#f4f1ea", showocean=True, oceancolor="#e6eef5",
                    showcoastlines=True, coastlinecolor="#888888",
                    lataxis_range=r["geo.lataxis.range"], lonaxis_range=r["geo.lonaxis.range"])
    fig.update_layout(title=dict(text=title, x=0.01, y=0.975, yanchor="top"),
                      margin=dict(l=10, r=10, t=135, b=75), height=720)


def _menu(fig, label: str, buttons: list, note: str) -> None:
    """Drop-down above the map with a visible label, and an explanation under the map."""
    fig.update_layout(
        updatemenus=[dict(type="dropdown", buttons=buttons, x=0.0, y=1.0, xanchor="left", yanchor="bottom")],
        annotations=[
            dict(text=f"<b>{label}</b>", xref="paper", yref="paper", x=0.0, y=1.075, xanchor="left",
                 yanchor="bottom", showarrow=False, font=dict(size=13)),
            dict(text=note, xref="paper", yref="paper", x=0.0, y=-0.015, xanchor="left", yanchor="top",
                 showarrow=False, align="left", font=dict(size=12, color="#555555"))])


def _sizes(values, base: float = 7.0, k: float = 3.0, cap: float = 40.0):
    return np.minimum(base + k * np.sqrt(np.asarray(values, float)), cap)


def _country_trace(go, summary: pd.DataFrame, coords: pd.DataFrame):
    """Country names in a discreet grey, so that they do not compete with the markers."""
    countries = {KEY.match(k).group(2) for k in summary.city_key if KEY.match(k)}
    c = country_centroids(coords, countries)
    return go.Scattergeo(lat=c.lat, lon=c.lon, mode="text", text=c.country, hoverinfo="skip", showlegend=False,
                         textfont=dict(size=12, color="#9c9c9c"))


def activity_map(go, summary: pd.DataFrame, coords: pd.DataFrame):
    hover = [f"<b>{r.city_key}</b><br>episodes: {r.n_episodes} ({r.spike} spike, {r.short} short, {r.sustained} sustained)"
             f"<br>anomalous days: {r.n_days}<br>excess posts: {r.excess_posts:,.0f}<br>largest peak: x{r.max_ratio:.1f}"
             for r in summary.itertuples()]
    fig = go.Figure(_country_trace(go, summary, coords))
    fig.add_trace(go.Scattergeo(
        lat=summary.lat, lon=summary.lon, mode="markers", text=hover, hovertemplate="%{text}<extra></extra>",
        showlegend=False,
        marker=dict(size=_sizes(summary.n_episodes), color=np.log10(summary.excess_posts.clip(lower=1)),
                    colorscale="YlOrRd", showscale=True, colorbar=dict(title="log10(excess posts)"),
                    line=dict(width=0.6, color="#444444"), opacity=0.85)))
    _geo(fig, "Activity anomalies per city (size = number of episodes, colour = excess posts)", summary)
    return fig


def _line_xy(rows):
    lat, lon = [], []
    for r in rows.itertuples():
        lat += [r.lat_a, r.lat_b, None]
        lon += [r.lon_a, r.lon_b, None]
    return lat, lon


LAG_COLOR, SAME_COLOR = "#e8710a", "#2f6db5"


def correlation_map(go, summary: pd.DataFrame, pairs: pd.DataFrame, coords: pd.DataFrame):
    """Cities + lines between correlated pairs; a drop-down shows all links, only 'follows' or only 'same day'."""
    fig = go.Figure(_country_trace(go, summary, coords))
    fig.add_trace(go.Scattergeo(
        lat=summary.lat, lon=summary.lon, mode="markers", name="cities", hoverinfo="skip", showlegend=False,
        marker=dict(size=_sizes(summary.n_episodes, base=5, k=1.5, cap=20), color="#444444", opacity=0.75)))
    groups = [None, None]                           # group of every trace: None = always shown
    if len(pairs):
        edges = np.quantile(pairs.best_corr, [0, 1 / 3, 2 / 3, 1])
        bucket = np.clip(np.searchsorted(edges, pairs.best_corr, side="right") - 1, 0, 2)
        for grp, color, label in (("lag", LAG_COLOR, "one city follows the other"), ("same", SAME_COLOR, "same day")):
            lagged = grp == "lag"
            for b, width in enumerate((1.2, 2.8, 5.0)):
                rows = pairs[(bucket == b) & ((pairs.best_lag_days != 0) == lagged)]
                if len(rows):
                    lat, lon = _line_xy(rows)
                    fig.add_trace(go.Scattergeo(lat=lat, lon=lon, mode="lines", hoverinfo="skip",
                                                line=dict(width=width, color=color), opacity=0.75,
                                                name=label, legendgroup=grp, showlegend=(b == 0)))
                    groups.append(grp)
            sub = pairs[(pairs.best_lag_days != 0) == lagged]
            if len(sub):
                text = [f"<b>{r.city_a}</b> - <b>{r.city_b}</b><br>correlation {r.best_corr:.2f}"
                        + (f"<br>{r.leader} first, by {abs(int(r.best_lag_days))} day(s)" if r.leader else "<br>same day")
                        for r in sub.itertuples()]
                fig.add_trace(go.Scattergeo(
                    lat=(sub.lat_a + sub.lat_b) / 2, lon=(sub.lon_a + sub.lon_b) / 2, mode="markers", text=text,
                    hovertemplate="%{text}<extra></extra>", showlegend=False,
                    marker=dict(size=7, color=color, opacity=0.5)))
                groups.append(grp)
        buttons = []
        for label, show in (("All links", {"lag", "same"}), ("One city follows the other", {"lag"}), ("Same day", {"same"})):
            vis = [g is None or g in show for g in groups]
            buttons.append(dict(label=label, method="update", args=[{"visible": vis}]))
        _menu(fig, "Links shown:", buttons,
              "Menu: all links, only the pairs where one city follows the other by 1-3 days, or only same-day pairs. "
              "Line width = strength of the correlation. Hover on the middle dot of a line for the pair and which city reacts first.")
        fig.update_layout(legend=dict(x=1.0, y=0.5))
    _geo(fig, f"Correlated anomalies between cities ({len(pairs)} strongest links, width = correlation)", summary)
    return fig


def sync_map(go, summary: pd.DataFrame, sync: pd.DataFrame, onsets: pd.DataFrame, coords: pd.DataFrame, top: int = 12):
    """One map with a drop-down: each synchronized peak, cities coloured by hours after the first city (star)."""
    peaks = peak_ranking(sync, top)
    fig = go.Figure(_country_trace(go, summary, coords))
    fig.add_trace(go.Scattergeo(lat=summary.lat, lon=summary.lon, mode="markers", name="all cities", hoverinfo="skip",
                                showlegend=False, marker=dict(size=5, color="#bbbbbb", opacity=0.8)))
    n_fixed = len(fig.data)
    per_peak, titles, labels, views = [], [], [], []
    for p in peaks.itertuples():
        o = onsets[onsets.sync_id == p.sync_id].join(coords, on="city_key").dropna(subset=["lat", "lon"])
        o = o.sort_values("onset_utc")
        if o.empty:
            continue
        hover = [f"<b>{r.city_key}</b><br>onset {r.onset_utc} UTC<br>{r.hours_after_first:.0f} h after the first city"
                 for r in o.itertuples()]
        first = o.iloc[:1]
        fig.add_trace(go.Scattergeo(
            lat=o.lat, lon=o.lon, mode="markers", text=hover, hovertemplate="%{text}<extra></extra>",
            visible=False, showlegend=False,
            marker=dict(size=13, color=o.hours_after_first, colorscale="Viridis", reversescale=True, showscale=True,
                        colorbar=dict(title="hours after<br>first city"), line=dict(width=0.8, color="#222222"))))
        fig.add_trace(go.Scattergeo(
            lat=first.lat, lon=first.lon, mode="markers+text", visible=False, showlegend=False,
            text=[f"first: {k}" for k in first.city_key], textposition="top center",
            textfont=dict(size=12, color="#d62728"), hovertemplate="first city to react<extra></extra>",
            marker=dict(size=26, symbol="star", color="rgba(0,0,0,0)", line=dict(width=2, color="#d62728"))))
        per_peak.append(p)
        views.append(view_range(o))
        titles.append(f"#{p.sync_id}  {str(p.start)[:10]} to {str(p.end)[:10]}  "
                      f"({len(o)} of {p.n_cities} cities with a measured onset)")
        labels.append(f"#{p.sync_id}  |  {str(p.start)[:10]} to {str(p.end)[:10]}  |  {p.n_cities} cities")
    n_tr = n_fixed + 2 * len(per_peak)
    buttons = []
    for i, t in enumerate(titles):
        vis = [True] * n_fixed + [False] * (n_tr - n_fixed)
        vis[n_fixed + 2 * i] = vis[n_fixed + 2 * i + 1] = True
        buttons.append(dict(label=labels[i], method="update",
                            args=[{"visible": vis}, {"title.text": f"Synchronized peak {t}  (star = first city to react)", **views[i]}]))
    if per_peak:
        for tr in fig.data[n_fixed:n_fixed + 2]:
            tr.visible = True
        _menu(fig, "Synchronized peak:", buttons,
              "Each menu entry is a synchronized peak: a period when several cities had an activity anomaly at the same time "
              "(number, dates, number of cities); the largest peaks come first. Colour = hours after the first city to react "
              "(yellow = first, dark = last). Red star = first city. Some cities of a peak have no measurable onset "
              "and are not drawn (see the count in the title).")
    first_view = summary
    if per_peak:
        first_view = onsets[onsets.sync_id == per_peak[0].sync_id].join(coords, on="city_key").dropna(subset=["lat", "lon"])
    _geo(fig, f"Synchronized peak {titles[0]}  (star = first city to react)" if titles
         else "Synchronized peaks (no onsets available)", first_view)
    return fig


def ego_map(go, summary: pd.DataFrame, corr: pd.DataFrame, coords: pd.DataFrame, top_links: int = 8, min_corr: float = 0.2):
    """One city at a time (drop-down): the cities most correlated with it, linked to it like a star.
    Orange = the partner follows (or leads) by 1-3 days, blue = same day."""
    c = corr[corr.best_corr >= min_corr].copy()
    c["leader"] = np.where(c.best_lag_days > 0, c.city_a, np.where(c.best_lag_days < 0, c.city_b, ""))
    long = pd.concat([
        c.assign(center=c.city_a, other=c.city_b),
        c.assign(center=c.city_b, other=c.city_a)], ignore_index=True)
    long = long.join(coords.add_suffix("_c"), on="center").join(coords.add_suffix("_o"), on="other")
    long = long.dropna(subset=["lat_c", "lon_c", "lat_o", "lon_o"])
    order = long.groupby("center").size().sort_values(ascending=False).index
    fig = go.Figure(_country_trace(go, summary, coords))
    fig.add_trace(go.Scattergeo(lat=summary.lat, lon=summary.lon, mode="markers", hoverinfo="skip", showlegend=False,
                                marker=dict(size=5, color="#bbbbbb", opacity=0.8)))
    n_fixed = len(fig.data)
    titles, views = [], []
    for city in order:
        rows = long[long.center == city].sort_values("best_corr", ascending=False).head(top_links)
        lagged = rows[rows.best_lag_days != 0]
        same = rows[rows.best_lag_days == 0]

        def xy(df):
            lat, lon = [], []
            for r in df.itertuples():
                lat += [r.lat_c, r.lat_o, None]
                lon += [r.lon_c, r.lon_o, None]
            return lat, lon

        for df, color in ((lagged, LAG_COLOR), (same, SAME_COLOR)):
            lat, lon = xy(df)
            fig.add_trace(go.Scattergeo(lat=lat, lon=lon, mode="lines", hoverinfo="skip", showlegend=False,
                                        visible=False, line=dict(width=2.5, color=color), opacity=0.8))
        text = []
        for r in rows.itertuples():
            when = "same day" if r.best_lag_days == 0 else (
                f"{city} first, {r.other} follows after {abs(int(r.best_lag_days))} day(s)" if r.leader == city
                else f"{r.other} first, {city} follows after {abs(int(r.best_lag_days))} day(s)")
            text.append(f"<b>{r.other}</b><br>correlation with {city}: {r.best_corr:.2f}<br>{when}")
        fig.add_trace(go.Scattergeo(
            lat=rows.lat_o, lon=rows.lon_o, mode="markers", text=text, hovertemplate="%{text}<extra></extra>",
            visible=False, showlegend=False,
            marker=dict(size=9 + 14 * rows.best_corr, line=dict(width=0.8, color="#222222"),
                        color=[LAG_COLOR if d != 0 else SAME_COLOR for d in rows.best_lag_days])))
        fig.add_trace(go.Scattergeo(
            lat=[rows.lat_c.iloc[0]], lon=[rows.lon_c.iloc[0]], mode="markers+text", text=[city], textposition="top center",
            textfont=dict(size=12, color="#222222"), hoverinfo="skip", visible=False, showlegend=False,
            marker=dict(size=22, symbol="star", color="rgba(0,0,0,0)", line=dict(width=2, color="#d62728"))))
        titles.append(city)
        views.append(view_range(pd.concat([rows[["lat_o", "lon_o"]].set_axis(["lat", "lon"], axis=1),
                                           rows[["lat_c", "lon_c"]].head(1).set_axis(["lat", "lon"], axis=1)])))
    n_tr = n_fixed + 4 * len(titles)
    buttons = []
    for i, t in enumerate(titles):
        vis = [True] * n_fixed + [False] * (n_tr - n_fixed)
        for j in range(4):
            vis[n_fixed + 4 * i + j] = True
        buttons.append(dict(label=t, method="update",
                            args=[{"visible": vis}, {"title.text": f"Cities most correlated with {t} (orange = by 1-3 days, blue = same day)",
                                                      **views[i]}]))
    if titles:
        for tr in fig.data[n_fixed:n_fixed + 4]:
            tr.visible = True
        _menu(fig, "City:", buttons,
              "Pick a city in the menu (red star): the map shows the cities whose anomalies are most correlated with it. "
              "Orange = the two cities react 1-3 days apart (hover to see which one is first), blue = same day. "
              "Dot size = strength of the correlation.")
    _geo(fig, f"Cities most correlated with {titles[0]} (orange = by 1-3 days, blue = same day)" if titles
         else "No correlated pairs", summary if not titles else
         pd.concat([long[long.center == titles[0]].head(top_links)[["lat_o", "lon_o"]].set_axis(["lat", "lon"], axis=1),
                    long[long.center == titles[0]][["lat_c", "lon_c"]].head(1).set_axis(["lat", "lon"], axis=1)]))
    return fig


# ------------------------------------------------------------------------------------------ driver
def save(fig, path: Path, offline: bool, png: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs=True if offline else "cdn", full_html=True)
    if png:
        try:
            fig.write_image(str(path.with_suffix(".png")), scale=2)
        except Exception as e:                                   # kaleido missing or no browser
            print(f"png skipped for {path.name}: {e}")


def run(args) -> dict:
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise SystemExit("plotly is required: pip install plotly")
    eps = load_table(args.anomalies, "anomaly_episodes")
    if eps is None or eps.empty:
        raise SystemExit(f"No anomaly_episodes in {args.anomalies}: run anomaly_peaks.py first")
    sync = load_table(args.anomalies, "synchronized_peaks")
    onsets = load_table(args.anomalies, "sync_onsets")
    corr = load_table(args.anomalies, "city_pair_correlation")
    coords = load_coords(args.coords)
    summary, missing = with_coords(city_summary(eps), coords)
    if missing:
        print(f"cities without a position (add them to {args.coords}): {', '.join(missing)}")
    figdir, out = Path(args.figdir), {}
    out["activity"] = activity_map(go, summary, coords)
    save(out["activity"], figdir / "map_anomaly_activity.html", args.offline, args.png)
    if corr is not None and len(corr):
        pairs = strongest_pairs(corr, coords, args.min_corr, args.top_pairs, args.per_city)
        out["correlation"] = correlation_map(go, summary, pairs, coords)
        save(out["correlation"], figdir / "map_city_correlation.html", args.offline, args.png)
        out["ego"] = ego_map(go, summary, corr, coords, args.top_links, 0.2)
        save(out["ego"], figdir / "map_city_links.html", args.offline, args.png)
    else:
        print("no city_pair_correlation: correlation map skipped")
    if sync is not None and len(sync) and onsets is not None and len(onsets):
        out["sync"] = sync_map(go, summary, sync, onsets, coords, args.top_peaks)
        save(out["sync"], figdir / "map_synchronized_peaks.html", args.offline, args.png)
    else:
        print("no sync_onsets: synchronized-peak map skipped (run job5 with --hourly, then anomaly_peaks.py)")
    print(f"maps written to {figdir}: {', '.join(sorted(p.name for p in figdir.glob('map_*.html')))}")
    return out


# ------------------------------------------------------------------------------------------ selftest
class _Stub:
    """Records what is drawn when plotly is not installed (checks the logic, not the plotly API)."""
    class Trace(dict):
        def __init__(self, **kw):
            super().__init__(kw)
            self.visible = kw.get("visible", True)

    class Figure:
        def __init__(self, trace=None):
            self.data = [trace] if trace is not None else []
            self.layout = {}

        def add_trace(self, t):
            self.data.append(t)

        def update_layout(self, **kw):
            self.layout.update(kw)

        def update_geos(self, **kw):
            self.layout["geo"] = kw

        def write_html(self, path, **kw):
            Path(path).write_text("<html></html>")


def selftest() -> None:
    try:
        import plotly.graph_objects as go
        stub = False
    except ImportError:
        go, stub = types_module(), True
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(3)
    cities = [("Moscow", "Russia", 55.76, 37.62), ("Kazan", "Russia", 55.79, 49.12), ("Kyiv", "Ukraine", 50.45, 30.52),
              ("Minsk", "Belarus", 53.90, 27.57), ("Nowhere", "Russia", np.nan, np.nan)]
    pd.DataFrame(cities, columns=["city", "country", "lat", "lon"]).dropna().to_csv(tmp / "coords.csv", index=False)
    keys = [f"{c} ({k})" for c, k, *_ in cities]
    eps = pd.DataFrame({"city_key": [keys[0]] * 3 + [keys[1]] * 2 + [keys[2], keys[4]],
                        "n_days": [1, 3, 9, 1, 2, 1, 1], "excess_posts": [100, 400, 5000, 50, 80, 30, 10],
                        "peak_ratio": [2.0, 3.0, 5.0, 2.5, 2.0, 2.0, 2.0],
                        "shape": ["spike", "short", "sustained", "spike", "spike", "spike", "spike"]})
    for name, df in (("anomaly_episodes", eps),
                     ("city_pair_correlation", pd.DataFrame({"city_a": [keys[0], keys[0], keys[2]],
                                                              "city_b": [keys[1], keys[2], keys[3]],
                                                              "corr_lag0": [.5, .4, .3], "best_lag_days": [1, 0, -2],
                                                              "best_corr": [.6, .4, .35]})),
                     ("synchronized_peaks", pd.DataFrame({"sync_id": [1, 2], "start": ["2022-01-01", "2022-03-01"],
                                                          "end": ["2022-01-02", "2022-03-01"], "n_cities": [3, 2],
                                                          "excess_posts": [900, 100]})),
                     ("sync_onsets", pd.DataFrame({"sync_id": [1, 1, 1, 2, 2], "city_key": [keys[2], keys[0], keys[1], keys[0], keys[3]],
                                                   "onset_utc": pd.to_datetime(["2022-01-01 03:00", "2022-01-01 07:00", "2022-01-01 20:00",
                                                                                "2022-03-01 10:00", "2022-03-01 12:00"]),
                                                   "hours_after_first": [0.0, 4.0, 17.0, 0.0, 2.0]}))):
        (tmp / "a" / name).mkdir(parents=True, exist_ok=True)
        df.to_csv(tmp / "a" / name / f"{name}.csv", index=False)
    # data preparation
    coords = load_coords(tmp / "coords.csv")
    summary, missing = with_coords(city_summary(eps), coords)
    assert missing == [keys[4]] and len(summary) == 3, (missing, summary)
    m = summary.set_index("city_key").loc[keys[0]]
    assert m.n_episodes == 3 and m.spike == 1 and m.short == 1 and m.sustained == 1 and m.excess_posts == 5500
    pairs = strongest_pairs(pd.read_csv(tmp / "a" / "city_pair_correlation" / "city_pair_correlation.csv"), coords, 0.3)
    lead = pairs.set_index(["city_a", "city_b"]).leader
    assert lead[(keys[0], keys[1])] == keys[0]          # lag +1: Kazan follows Moscow
    assert lead[(keys[2], keys[3])] == keys[3]          # lag -2: Kyiv follows Minsk
    assert lead[(keys[0], keys[2])] == ""               # same day
    capped = strongest_pairs(pd.read_csv(tmp / "a" / "city_pair_correlation" / "city_pair_correlation.csv"), coords, 0.3, 50, 1)
    assert len(capped) == 2 and set(capped.city_b) == {keys[1], keys[3]}, capped     # city 0 keeps one link only
    cc = country_centroids(coords, {"Russia"})
    assert list(cc.country) == ["Russia"] and abs(cc.lat[0] - 55.775) < 1e-6
    assert list(peak_ranking(pd.read_csv(tmp / "a" / "synchronized_peaks" / "synchronized_peaks.csv")).sync_id) == [1, 2]
    # maps
    if stub:
        sys.modules["plotly"] = types_module(pkg=True)
        sys.modules["plotly.graph_objects"] = go
    args = argparse.Namespace(anomalies=str(tmp / "a"), coords=str(tmp / "coords.csv"), figdir=str(tmp / "fig"),
                              min_corr=0.3, top_pairs=50, per_city=4, top_links=8, top_peaks=5, offline=False, png=False)
    out = run(args)
    assert set(out) == {"activity", "correlation", "ego", "sync"}
    for f in ("map_anomaly_activity.html", "map_city_correlation.html", "map_city_links.html", "map_synchronized_peaks.html"):
        assert (tmp / "fig" / f).exists(), f
    assert len(out["sync"].data) == 2 + 2 * 2           # country names + background + (cities, star) per peak
    assert len(out["ego"].data) == 2 + 4 * 4, len(out["ego"].data)       # 4 cities have partners, 4 traces each
    assert len(out["correlation"].data) >= 3 and len(out["activity"].data) == 2
    print("selftest OK" + (" (plotly not installed: figure objects stubbed, API calls not exercised)" if stub else ""))


def types_module(pkg: bool = False):
    import types
    m = types.ModuleType("plotly" if pkg else "plotly.graph_objects")
    if not pkg:
        m.Figure = _Stub.Figure
        m.Scattergeo = _Stub.Trace
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anomalies", default="bigdata_pipeline/analysis/anomalies")
    ap.add_argument("--coords", default="bigdata_pipeline/analysis/city_coordinates/city_coordinates.csv")
    ap.add_argument("--figdir", default="bigdata_pipeline/analysis/figures/anomalies")
    ap.add_argument("--min-corr", type=float, default=0.35, help="minimum correlation of a pair drawn on the map")
    ap.add_argument("--top-pairs", type=int, default=80, help="maximum number of pairs drawn")
    ap.add_argument("--per-city", type=int, default=4, help="maximum number of links kept for one city")
    ap.add_argument("--top-links", type=int, default=8, help="partners shown around the selected city (city-by-city map)")
    ap.add_argument("--top-peaks", type=int, default=12, help="number of synchronized peaks in the drop-down")
    ap.add_argument("--offline", action="store_true", help="embed plotly.js in each HTML file (about 4 MB each)")
    ap.add_argument("--png", action="store_true", help="also write PNG files (needs: pip install kaleido)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == "__main__":
    main()