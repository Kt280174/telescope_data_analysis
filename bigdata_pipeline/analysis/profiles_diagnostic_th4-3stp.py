#!/usr/bin/env python3
"""Diagnostic: can non-channel rows (bots, personal profiles) be used for the geographic analysis?

For every row of Source_Channel_Metadata the script decides its kind (channel = has a title,
bot = bot flag, profile = anything else) and counts, per kind:
  - how many have a filled lang_code (and the most frequent codes),
  - how many have a 24-hour activity profile in Temporal_Channel_Metadata,
  - how many appear in the forward graph (as origin and as destination).
Only counts are printed. No per-profile output is written.

Usage:
  python3 bigdata_pipeline/analysis/profile_diagnostic.py \
    --metadata telescope_files/Source_Channel_Metadata.csv \
    --temporal telescope_files/Temporal_Channel_Metadata.csv \
    --gold telescope_pipeline_files/gold
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
csv.field_size_limit(sys.maxsize)


def classify(path: str):
    """Return (kind per id, lang_code counter per kind)."""
    kind: dict[int, str] = {}
    langs = {"channel": Counter(), "bot": Counter(), "profile": Counter()}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                obj = json.loads(row["data"])
                cid = int(row["id"])
            except (ValueError, KeyError, TypeError):
                continue
            if obj.get("title"):
                k = "channel"
            elif obj.get("bot"):
                k = "bot"
            else:
                k = "profile"
            kind[cid] = k
            langs[k][obj.get("lang_code") or ""] += 1
    return kind, langs


def temporal_ids(path: str, min_posts: int):
    df = pd.read_csv(path, dtype=str)
    ids = df.iloc[:, 0].astype(str).str.extract(r"(\d+)")[0]
    vals = df.iloc[:, 1:25].apply(pd.to_numeric, errors="coerce").fillna(0)
    any_ids = set(ids.dropna().astype("int64"))
    enough = ids.notna() & (vals.sum(axis=1) >= min_posts)
    return any_ids, set(ids[enough].astype("int64"))


def pct(a: int, b: int) -> str:
    return f"{a:>9,}  ({100 * a / b:5.1f} %)" if b else f"{a:>9,}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--temporal", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--min-posts", type=int, default=100)
    args = ap.parse_args()

    kind, langs = classify(args.metadata)
    ids_by_kind = {k: {i for i, v in kind.items() if v == k} for k in langs}
    print(f"rows read: {len(kind):,}")

    t_any, t_enough = temporal_ids(args.temporal, args.min_posts)

    from community_detection_v2 import load_edges
    edges = load_edges(args.gold)
    origins = set(edges["origin_channel"].unique())
    dests = set(edges["dest_channel"].unique())
    # out-degree towards distinct channels, used to see how much a row shares
    sharing = edges.groupby("dest_channel")["origin_channel"].nunique()

    for k, ids in ids_by_kind.items():
        n = len(ids)
        print(f"\n== {k} ({n:,}) ==")
        filled = n - langs[k][""]
        print("lang_code filled      ", pct(filled, n))
        top = ", ".join(f"{c}:{v:,}" for c, v in langs[k].most_common(8) if c)
        print("  most frequent codes ", top or "-")
        print("temporal profile      ", pct(len(ids & t_any), n))
        print(f"  with >= {args.min_posts} posts   ", pct(len(ids & t_enough), n))
        print("in graph as origin    ", pct(len(ids & origins), n))
        print("in graph as dest      ", pct(len(ids & dests), n))
        both = ids & t_enough & (origins | dests)
        print("usable (activity+graph)", pct(len(both), n))
        s = sharing.reindex(list(ids & dests)).dropna()
        if len(s):
            print("distinct sources shared by a dest: median "
                  f"{s.median():.0f}, share with >=3: {100 * (s >= 3).mean():.1f} %")


if __name__ == "__main__":
    main()