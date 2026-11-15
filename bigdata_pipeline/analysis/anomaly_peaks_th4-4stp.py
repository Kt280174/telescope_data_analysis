#!/usr/bin/env python3
"""
Theme 4 - temporal anomalies (activity peaks) per city, segmentation, cross-city correlation, synchronized peaks.

Input : gold/city_daily (written by job5_city_daily.py): one row per (city, UTC day).
Steps
  1. Daily series per city (missing days = 0). Metric: n_posts by default.
  2. Anomaly score per city and day: log1p(count) minus its weekday effect, compared with the median of
     the previous `--window` days, divided by a robust spread (1.4826 * MAD of the previous deviations, with
     a floor). A day is anomalous when score >= --z and the raw count >= --min-count.
  3. Segmentation: consecutive anomalous days (gap <= --gap) form an episode, labelled spike (1-2 days),
     short (3-7 days) or sustained (more than 7 days).
  4. Correlation: (a) synchronized peaks = groups of days where at least --min-cities cities are anomalous together;
     the city whose episode starts first is the lead city; days where more than --global-share of the cities
     are anomalous are flagged `global` (often a collection gap or a platform-wide effect, not a city-level pattern);
     (b) lagged correlation of the positive anomaly scores between the busiest cities.

Outputs under --out (one folder per table, as the other analysis scripts) and figures under --figdir.
Usage:
  python3 bigdata_pipeline/analysis/anomaly_peaks.py \
    --gold telescope_pipeline_files/gold \
    --out bigdata_pipeline/analysis/anomalies \
    --figdir bigdata_pipeline/analysis/figures/anomalies
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

SIGMA_FLOOR = 0.15          # minimum spread on the log scale (about 16 % of day-to-day change)


# ------------------------------------------------------------------------------------------ loading
def load_city_daily(gold) -> pd.DataFrame:
    gold = Path(gold)
    pq, csvdir = gold / "city_daily", gold / "city_daily_csv"
    df = None
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except ImportError:
            df = None
    if df is None:
        parts = sorted(csvdir.glob("*.csv")) if csvdir.exists() else []
        if not parts:
            raise SystemExit(f"No city_daily found: expected {pq} (Parquet) or {csvdir}/*.csv (run job5_city_daily.py)")
        df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["city_key"] = df["city"].astype(str) + " (" + df["country"].astype(str) + ")"
    return df


def daily_matrix(df: pd.DataFrame, metric: str = "n_posts", min_days: int = 90, min_total: int = 500) -> pd.DataFrame:
    """Days x cities matrix; missing days are 0. Cities with too little history or volume are dropped."""
    m = df.pivot_table(index="date", columns="city_key", values=metric, aggfunc="sum")
    m = m.reindex(pd.date_range(m.index.min(), m.index.max(), freq="D")).fillna(0.0)
    keep = ((m > 0).sum() >= min_days) & (m.sum() >= min_total)
    return m.loc[:, keep]


# ------------------------------------------------------------------------------------------ scoring
def anomaly_scores(m: pd.DataFrame, window: int = 56, min_periods: int = 28):
    """Robust score per city and day, plus the expected (baseline) count. Both are days x cities."""
    y = np.log1p(m)
    weekday = y.groupby(y.index.dayofweek).transform("median") - y.median()
    y = y - weekday
    base = y.shift(1).rolling(window, min_periods=min_periods).median()
    dev = (y - base).abs()
    sigma = (1.4826 * dev.shift(1).rolling(window, min_periods=min_periods).median()).clip(lower=SIGMA_FLOOR)
    z = (y - base) / sigma
    expected = np.expm1(base + weekday).clip(lower=0)
    return z, expected


def anomaly_mask(m: pd.DataFrame, z: pd.DataFrame, z_thr: float = 4.0, min_count: float = 20.0) -> pd.DataFrame:
    return (z >= z_thr) & (m >= min_count)


# ------------------------------------------------------------------------------------------ episodes
def shape_of(n_days: int) -> str:
    return "spike" if n_days <= 2 else ("short" if n_days <= 7 else "sustained")


def episodes(m: pd.DataFrame, z: pd.DataFrame, expected: pd.DataFrame, mask: pd.DataFrame, gap: int = 1) -> pd.DataFrame:
    """Group consecutive anomalous days of each city (gaps up to `gap` days are bridged)."""
    rows = []
    for city in m.columns:
        days = np.flatnonzero(mask[city].to_numpy())
        if len(days) == 0:
            continue
        splits = np.flatnonzero(np.diff(days) > gap + 1) + 1
        for grp in np.split(days, splits):
            a, b = grp[0], grp[-1]
            sl = slice(a, b + 1)
            cnt, exp = m[city].to_numpy()[sl], expected[city].to_numpy()[sl]
            zz = z[city].to_numpy()[sl]
            k = int(np.nanargmax(zz))
            peak_exp = float(exp[k])
            rows.append({
                "city_key": city, "start": m.index[a], "end": m.index[b], "n_days": int(b - a + 1),
                "peak_date": m.index[a + k], "peak_count": float(cnt[k]), "expected_at_peak": round(peak_exp, 1),
                "peak_ratio": round(float(cnt[k]) / max(peak_exp, 1.0), 2), "max_score": round(float(zz[k]), 2),
                "excess_posts": round(float(np.clip(cnt - exp, 0, None).sum()), 1), "shape": shape_of(int(b - a + 1)),
            })
    cols = ["city_key", "start", "end", "n_days", "peak_date", "peak_count", "expected_at_peak", "peak_ratio",
            "max_score", "excess_posts", "shape"]
    return pd.DataFrame(rows, columns=cols).sort_values(["start", "city_key"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------ synchronized peaks
def synchronized_peaks(mask: pd.DataFrame, eps: pd.DataFrame, min_cities: int = 3, gap: int = 1,
               global_share: float = 0.5) -> pd.DataFrame:
    """Days where several cities are anomalous together, grouped into synchronized peaks with a lead city."""
    n = mask.sum(axis=1).to_numpy()
    days = np.flatnonzero(n >= min_cities)
    cols = ["sync_id", "start", "end", "n_days", "n_cities", "max_share_of_cities", "global", "lead_city",
            "n_lead_ties", "lead_lag_days", "excess_posts", "top_cities"]
    if len(days) == 0:
        return pd.DataFrame(columns=cols)
    rows, members = [], {}
    splits = np.flatnonzero(np.diff(days) > gap + 1) + 1
    for i, grp in enumerate(np.split(days, splits), 1):
        start, end = mask.index[grp[0]], mask.index[grp[-1]]
        sub = mask.iloc[grp[0]:grp[-1] + 1]
        active = sub.columns[sub.any(axis=0)]
        e = eps[eps.city_key.isin(active) & (eps.end >= start) & (eps.start <= end)]
        first = e.groupby("city_key").start.min()
        score = e.groupby("city_key").max_score.max()
        first = first.to_frame("start").assign(score=score).sort_values(["start", "score"], ascending=[True, False])
        # several cities often start on the same day: the lead is then the one with the highest score,
        # and n_lead_ties tells how many cities share that first day (1 = a real first mover)
        lead = first.index[0] if len(first) else ""
        n_ties = int((first.start == first.start.iloc[0]).sum()) if len(first) else 0
        lag = int((first.start.iloc[-1] - first.start.iloc[0]).days) if len(first) else 0
        exc = e.groupby("city_key").excess_posts.sum().sort_values(ascending=False)
        members[i] = list(active)
        rows.append({
            "sync_id": i, "start": start, "end": end, "n_days": int((end - start).days + 1),
            "n_cities": len(active), "max_share_of_cities": round(float(n[grp].max() / mask.shape[1]), 3),
            "global": bool(n[grp].max() / mask.shape[1] > global_share), "lead_city": lead, "n_lead_ties": n_ties, "lead_lag_days": lag,
            "excess_posts": round(float(exc.sum()), 1), "top_cities": "; ".join(exc.index[:8]),
        })
    out = pd.DataFrame(rows, columns=cols)
    out.attrs["cities"] = members                # sync_id -> cities anomalous during the synchronized peak
    return out


# ------------------------------------------------------------------------------------------ hourly onsets
def load_city_hourly(gold):
    """city_hourly written by job5 --hourly, or None when it is not there."""
    gold = Path(gold)
    pq, csvdir = gold / "city_hourly", gold / "city_hourly_csv"
    df = None
    if pq.exists():
        try:
            df = pd.read_parquet(pq)
        except ImportError:
            df = None
    if df is None and csvdir.exists() and list(csvdir.glob("*.csv")):
        df = pd.concat((pd.read_csv(p) for p in sorted(csvdir.glob("*.csv"))), ignore_index=True)
    if df is None:
        return None
    df["hour"] = pd.to_datetime(df["hour"])
    df["city_key"] = df["city"].astype(str) + " (" + df["country"].astype(str) + ")"
    return df


def hourly_matrix(hdf: pd.DataFrame, cities) -> pd.DataFrame:
    m = hdf[hdf.city_key.isin(list(cities))].pivot_table(index="hour", columns="city_key", values="n_posts", aggfunc="sum")
    return m.reindex(pd.date_range(m.index.min().floor("D"), m.index.max().ceil("D"), freq="h")).fillna(0.0)


def onset_hour(s: pd.Series, start: pd.Timestamp, end: pd.Timestamp, k: float = 3.0, min_extra: float = 5.0,
               before_days: int = 1, after_days: int = 2):
    """First hour of the synchronized peak window where the city is above its usual level for two hours in a row.
    The usual level of an hour is the median of the same weekday and hour in the 4 previous weeks.
    Returns a Timestamp or None."""
    base = pd.concat([s.shift(24 * 7 * j) for j in range(1, 5)], axis=1).median(axis=1)
    thr = np.maximum(k * base, base + min_extra)
    lo = start - pd.Timedelta(days=before_days)
    hi = min(end, start + pd.Timedelta(days=after_days)) + pd.Timedelta(hours=23)
    seg, t = s.loc[lo:hi], thr.loc[lo:hi]
    hit = (seg >= t) & t.notna()
    hit = hit & hit.shift(-1, fill_value=False)
    return hit.index[hit.to_numpy()][0] if hit.any() else None


def sync_onsets(sync: pd.DataFrame, hdf: pd.DataFrame, k: float = 3.0, min_extra: float = 5.0) -> pd.DataFrame:
    """Onset hour of each city in each synchronized peak, and hours after the first city."""
    members = sync.attrs.get("cities", {})
    all_cities = sorted({c for v in members.values() for c in v})
    cols = ["sync_id", "city_key", "onset_utc", "hours_after_first"]
    if not all_cities:
        return pd.DataFrame(columns=cols)
    H = hourly_matrix(hdf, all_cities)
    rows = []
    for _, e in sync.iterrows():
        found = []
        for c in members.get(e.sync_id, []):
            if c in H.columns:
                t = onset_hour(H[c], e.start, e.end, k, min_extra)
                if t is not None:
                    found.append((c, t))
        found.sort(key=lambda x: x[1])
        for c, t in found:
            rows.append({"sync_id": int(e.sync_id), "city_key": c, "onset_utc": t,
                         "hours_after_first": round((t - found[0][1]).total_seconds() / 3600, 1)})
    return pd.DataFrame(rows, columns=cols)


def add_onset_columns(sync: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """Add the hourly lead city, the gap to the second city and the spread (hours) to the synchronized peaks table."""
    ev = sync.copy()
    lead, gap, spread, n = [], [], [], []
    for eid in ev.sync_id:
        o = onsets[onsets.sync_id == eid].sort_values("onset_utc")
        n.append(len(o))
        lead.append(o.city_key.iloc[0] if len(o) else "")
        gap.append(float(o.hours_after_first.iloc[1]) if len(o) > 1 else np.nan)
        spread.append(float(o.hours_after_first.iloc[-1]) if len(o) > 1 else np.nan)
    ev["n_cities_with_onset"], ev["lead_city_hourly"] = n, lead
    ev["lead_gap_hours"], ev["onset_spread_hours"] = gap, spread
    ev.attrs.update(sync.attrs)
    return ev


# ------------------------------------------------------------------------------------------ correlation
def lagged_correlation(z: pd.DataFrame, top: int = 40, max_lag: int = 3, min_corr: float = 0.2) -> pd.DataFrame:
    """Correlation of the positive anomaly scores between the `top` busiest cities, for lags -max_lag..max_lag.
    lag > 0 means city_b reacts `lag` days after city_a."""
    cols = list(z.columns[:top])
    X = np.clip(z[cols].fillna(0).to_numpy(float), 0, 10)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0) == 0, 1, X.std(axis=0))
    T = len(X)
    rows = []
    for a in range(len(cols)):
        for b in range(a + 1, len(cols)):
            best_lag, best = 0, -2.0
            c0 = float((X[:, a] * X[:, b]).mean())
            for lag in range(-max_lag, max_lag + 1):
                s_a, s_b = max(0, -lag), max(0, lag)
                n = T - abs(lag)
                c = float((X[s_a:s_a + n, a] * X[s_b:s_b + n, b]).mean())
                if c > best:
                    best, best_lag = c, lag
            if best >= min_corr:
                rows.append({"city_a": cols[a], "city_b": cols[b], "corr_lag0": round(c0, 3),
                             "best_lag_days": best_lag, "best_corr": round(best, 3)})
    out = pd.DataFrame(rows, columns=["city_a", "city_b", "corr_lag0", "best_lag_days", "best_corr"])
    return out.sort_values("best_corr", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------------------ driver
def write_table(out: Path, name: str, df: pd.DataFrame) -> None:
    d = Path(out) / name
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv", index=False)


def charts(figdir: Path, m: pd.DataFrame, mask: pd.DataFrame, sync: pd.DataFrame, eps: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir.mkdir(parents=True, exist_ok=True)
    # 1. number of cities anomalous each day, synchronized peaks shaded
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(mask.index, mask.sum(axis=1), color="#1f77b4", linewidth=0.8)
    for _, e in sync.iterrows():
        ax.axvspan(e.start, e.end + pd.Timedelta(days=1), color="#d62728" if e["global"] else "#ff7f0e", alpha=0.25)
    ax.set_title("Cities with an activity anomaly per day (shaded: synchronized peaks, red = global)")
    ax.set_ylabel("number of cities")
    fig.tight_layout()
    fig.savefig(figdir / "cities_anomalous_per_day.png", dpi=140)
    plt.close(fig)
    # 2. the busiest cities with their anomalous days marked
    cols = list(m.sum().sort_values(ascending=False).index[:6])
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 1.9 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        ax.plot(m.index, m[c], color="#555555", linewidth=0.7)
        hit = mask[c]
        ax.scatter(m.index[hit], m[c][hit], color="#d62728", s=10, zorder=3)
        ax.set_ylabel(c, rotation=0, ha="right", va="center", fontsize=8)
    np.atleast_1d(axes)[0].set_title("Daily posts of the busiest cities (red: anomalous days)")
    fig.tight_layout()
    fig.savefig(figdir / "top_cities_series.png", dpi=140)
    plt.close(fig)
    # 3. episodes per month and shape
    if len(eps):
        t = eps.assign(month=eps.start.dt.to_period("M").dt.to_timestamp()).groupby(["month", "shape"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(11, 3.8))
        t.plot.bar(stacked=True, ax=ax, width=0.9)
        ax.set_xticks(range(0, len(t), max(1, len(t) // 12)))
        ax.set_xticklabels([d.strftime("%Y-%m") for d in t.index[::max(1, len(t) // 12)]], rotation=45, ha="right")
        ax.set_title("Anomaly episodes started per month, by shape")
        fig.tight_layout()
        fig.savefig(figdir / "episodes_per_month.png", dpi=140)
        plt.close(fig)


def run(args) -> dict:
    df = load_city_daily(args.gold)
    m = daily_matrix(df, args.metric, args.min_days, args.min_total)
    if m.shape[1] == 0:
        raise SystemExit("No city has enough history; lower --min-days / --min-total")
    z, expected = anomaly_scores(m, args.window)
    mask = anomaly_mask(m, z, args.z, args.min_count)
    eps = episodes(m, z, expected, mask, args.gap)
    ev = synchronized_peaks(mask, eps, args.min_cities, args.gap, args.global_share)
    onsets = None
    hdf = None if args.no_hourly else load_city_hourly(args.gold)
    sens = None
    if hdf is not None and len(ev):
        onsets = sync_onsets(ev, hdf, args.onset_k, args.onset_extra)
        ev = add_onset_columns(ev, onsets)
        if args.sensitivity:
            # does the lead city survive a stricter and a looser onset threshold?
            leads = {}
            for k in [float(x) for x in args.sensitivity.split(",")]:
                o = sync_onsets(ev, hdf, k, args.onset_extra).sort_values("onset_utc")
                leads[f"lead_k{k:g}"] = o.groupby("sync_id").city_key.first()
            sens = pd.DataFrame(leads).reindex(ev.sync_id).fillna("")
            sens["stable"] = sens.nunique(axis=1) == 1
            sens = sens.reset_index()
            ev["lead_stable"] = ev.sync_id.map(sens.set_index("sync_id").stable)
    elif hdf is None:
        print("no city_hourly found (run job5_city_daily.py --hourly): onsets inside the day are skipped")
    order = m.sum().sort_values(ascending=False).index
    corr = lagged_correlation(z[order], args.top_corr, args.max_lag)
    summary = pd.DataFrame([
        ("cities analysed", m.shape[1]), ("days", len(m)), ("first day", m.index[0].date()), ("last day", m.index[-1].date()),
        ("anomalous city-days", int(mask.values.sum())), ("episodes", len(eps)),
        ("... spikes (1-2 days)", int((eps["shape"] == "spike").sum())),
        ("... short (3-7 days)", int((eps["shape"] == "short").sum())),
        ("... sustained (> 7 days)", int((eps["shape"] == "sustained").sum())),
        (f"synchronized peaks (>= {args.min_cities} cities together)", len(ev)),
        ("... flagged global", int(ev["global"].sum()) if len(ev) else 0),
    ], columns=["metric", "value"])
    out = Path(args.out)
    tables = [("anomaly_episodes", eps), ("synchronized_peaks", ev), ("city_pair_correlation", corr), ("anomaly_summary", summary)]
    if onsets is not None:
        tables.append(("sync_onsets", onsets))
    if sens is not None:
        tables.append(("lead_sensitivity", sens))
    res = {"onsets": onsets, "matrix": m, "z": z, "mask": mask, "episodes": eps, "sync": ev, "corr": corr, "summary": summary}
    for name, t in tables:
        write_table(out, name, t)
    if not args.no_charts:
        charts(Path(args.figdir), m, mask, ev, eps)
    print(summary.to_string(index=False))
    if len(ev):
        print("\nlargest synchronized peaks:")
        show = ["sync_id", "start", "end", "n_cities", "global", "lead_city", "n_lead_ties"]
        if "lead_city_hourly" in ev:
            show += ["lead_city_hourly", "lead_gap_hours", "onset_spread_hours"]
        if "lead_stable" in ev:
            show += ["lead_stable"]
        print(ev.sort_values("n_cities", ascending=False).head(10)[show].to_string(index=False))
    print(f"\ntables in {out}, figures in {args.figdir}")
    return res


# ------------------------------------------------------------------------------------------ selftest
def selftest() -> None:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2022-01-01", periods=240, freq="D")
    hours = pd.date_range(idx[0], periods=240 * 24, freq="h")
    onset_hour_of = {0: 10, 1: 14, 2: 16, 3: 15}     # hour of day at which each city starts to react
    rows_d, rows_h = [], []
    for i in range(8):
        lam = 60 + 40 * i
        wk = 1 + 0.2 * np.sin(2 * np.pi * hours.dayofweek / 7)
        h = rng.poisson(lam * wk / 24).astype(float)
        if i < 4:                                    # common synchronized peak: city 0 reacts on day 149, the others on day 150
            d0 = 149 if i == 0 else 150
            h[d0 * 24 + onset_hour_of[i]: (d0 + 1) * 24] *= 10
            h[(d0 + 1) * 24: (d0 + 2) * 24] *= 4
        if i == 7:                                   # an isolated spike in one city only
            h[80 * 24: 81 * 24] *= 6
        daily = pd.Series(h, index=hours).resample("D").sum()
        for t, v in zip(hours, h):
            rows_h.append({"country": "X", "city": f"C{i}", "hour": t, "n_posts": v})
        for d, v in daily.items():
            rows_d.append({"country": "X", "city": f"C{i}", "date": d, "n_posts": v})
    tmp = Path(tempfile.mkdtemp())
    (tmp / "gold" / "city_daily_csv").mkdir(parents=True)
    (tmp / "gold" / "city_hourly_csv").mkdir(parents=True)
    pd.DataFrame(rows_d).to_csv(tmp / "gold" / "city_daily_csv" / "p.csv", index=False)
    pd.DataFrame(rows_h).to_csv(tmp / "gold" / "city_hourly_csv" / "p.csv", index=False)
    args = argparse.Namespace(gold=str(tmp / "gold"), out=str(tmp / "out"), figdir=str(tmp / "fig"), metric="n_posts",
                              min_days=30, min_total=100, window=56, z=4.0, min_count=20, gap=1, min_cities=3,
                              global_share=0.5, top_corr=8, max_lag=3,
                              no_charts=False, no_hourly=False, onset_k=3.0, onset_extra=5.0,
                              sensitivity="2.5,3,4")
    r = run(args)
    ev, eps = r["sync"], r["episodes"]
    assert len(ev) == 1, ev
    e = ev.iloc[0]
    assert e.start <= idx[150] <= e.end and e.n_cities >= 4 and e.lead_city == "C0 (X)" and not e["global"], e
    assert e.lead_lag_days == 1 and e.n_lead_ties == 1
    assert e.lead_city_hourly == "C0 (X)" and e.n_cities_with_onset == 4, e
    assert bool(e.lead_stable) and (tmp / "out" / "lead_sensitivity" / "lead_sensitivity.csv").exists()
    on = r["onsets"].set_index("city_key")
    assert on.loc["C0 (X)", "onset_utc"] == idx[149] + pd.Timedelta(hours=10)
    assert on.loc["C1 (X)", "hours_after_first"] == 28.0 and e.lead_gap_hours == 28.0 and e.onset_spread_hours == 30.0
    iso = eps[eps.city_key == "C7 (X)"]
    assert len(iso) >= 1 and (iso.peak_date == idx[80]).any() and (iso["shape"] == "spike").any()
    assert not ((ev.start <= idx[80]) & (ev.end >= idx[80])).any()      # a single-city spike is not a synchronized peak
    assert not r["corr"].empty and (r["corr"].iloc[0].best_corr > 0.3)
    for name in ("anomaly_episodes", "synchronized_peaks", "city_pair_correlation", "anomaly_summary", "sync_onsets"):
        assert (tmp / "out" / name / f"{name}.csv").exists(), name
    assert (tmp / "fig" / "cities_anomalous_per_day.png").exists()
    print("selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="telescope_pipeline_files/gold")
    ap.add_argument("--out", default="bigdata_pipeline/analysis/anomalies")
    ap.add_argument("--figdir", default="bigdata_pipeline/analysis/figures/anomalies")
    ap.add_argument("--metric", default="n_posts", choices=["n_posts", "n_forwarded_posts", "views", "n_channels"])
    ap.add_argument("--min-days", type=int, default=90, help="minimum number of active days for a city")
    ap.add_argument("--min-total", type=int, default=500, help="minimum total of the metric for a city")
    ap.add_argument("--window", type=int, default=56, help="days of history used as baseline")
    ap.add_argument("--z", type=float, default=4.0, help="anomaly score threshold")
    ap.add_argument("--min-count", type=float, default=20.0, help="minimum raw count of an anomalous day")
    ap.add_argument("--gap", type=int, default=1, help="days that can be bridged inside an episode or synchronized peak")
    ap.add_argument("--min-cities", type=int, default=3, help="cities anomalous together to make a synchronized peak")
    ap.add_argument("--global-share", type=float, default=0.5, help="share of cities above which a synchronized peak is `global`")
    ap.add_argument("--top-corr", type=int, default=40, help="busiest cities used for the lagged correlation")
    ap.add_argument("--max-lag", type=int, default=3)
    ap.add_argument("--no-hourly", action="store_true", help="ignore city_hourly even if it exists")
    ap.add_argument("--onset-k", type=float, default=3.0, help="onset: hourly count above k times the usual level")
    ap.add_argument("--onset-extra", type=float, default=5.0, help="onset: ... and at least this many posts above it")
    ap.add_argument("--sensitivity", default=None,
                    help="comma list of onset-k values (e.g. 2.5,3,4): checks that the hourly lead city does not change")
    ap.add_argument("--no-charts", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    run(args)


if __name__ == "__main__":
    main()