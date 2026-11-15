#!/usr/bin/env python3
"""
JOB 5 - DAILY ACTIVITY PER CITY.

Input : <out>/clean_dedup (or clean)  +  channel_geo.csv produced by geo_cities.py
Output: <out>/gold/city_daily   (Parquet), one row per (city, UTC day):

  country, city, date, n_posts, n_forwarded_posts, views, n_channels

  n_posts            messages published that day by the channels located in the city
  n_forwarded_posts  of those, messages forwarded from a public channel
  views              sum of the views of those messages
  n_channels         distinct channels of the city that published that day

With --hourly, gold/city_hourly (country, city, hour, n_posts) is written too: it is used to time the
onset of each key event inside the day.

Only channels with exactly one candidate city are used (n_cities == 1), so a channel never counts twice.
Days are UTC days (the `ts` column is a Unix timestamp).

Usage: python3 job5_city_daily.py --out ~/telescope_v2 \
         --geo bigdata_pipeline/analysis/geo/channel_geo/channel_geo.csv [--csv]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--clean", type=Path, default=None)
    ap.add_argument("--geo", type=Path, required=True, help="channel_geo.csv written by geo_cities.py")
    ap.add_argument("--min-date", default="2015-01-01", help="drop messages dated before this day")
    ap.add_argument("--max-date", default="2026-12-31", help="drop messages dated after this day")
    ap.add_argument("--hourly", action="store_true", help="also write gold/city_hourly (city, UTC hour, n_posts)")
    ap.add_argument("--csv", action="store_true", help="also write a single CSV file")
    args = ap.parse_args()
    from spark_session import default_clean
    clean = args.clean or default_clean(args.out)

    from pyspark.sql import functions as F
    from spark_session import get_spark
    spark = get_spark("telescope-job5-city-daily")

    geo = (spark.read.option("header", True).csv(str(args.geo))
           .select(F.col("channel_id").cast("long").alias("channel_id"), "country", "city",
                   F.col("n_cities").cast("int").alias("n_cities"))
           .where((F.col("n_cities") == 1) & F.col("channel_id").isNotNull())
           .dropDuplicates(["channel_id"]))
    print(f"channels with a single city: {geo.count():,}")

    msgs = (spark.read.parquet(str(clean))
            .where(F.col("dest_channel").isNotNull() & F.col("ts").isNotNull())
            .select("dest_channel", "ts", "views", "kind")
            .withColumn("date", F.to_date(F.timestamp_seconds("ts")))
            .where((F.col("date") >= F.lit(args.min_date)) & (F.col("date") <= F.lit(args.max_date))))

    daily = (msgs.join(F.broadcast(geo), msgs.dest_channel == geo.channel_id)
             .groupBy("country", "city", "date")
             .agg(F.count(F.lit(1)).alias("n_posts"),
                  F.sum(F.when(F.col("kind") == "channel", 1).otherwise(0)).alias("n_forwarded_posts"),
                  F.coalesce(F.sum("views"), F.lit(0)).alias("views"),
                  F.countDistinct("dest_channel").alias("n_channels")))

    if args.hourly:
        hourly = (msgs.join(F.broadcast(geo), msgs.dest_channel == geo.channel_id)
                  .withColumn("hour", F.date_trunc("hour", F.timestamp_seconds("ts")))
                  .groupBy("country", "city", "hour")
                  .agg(F.count(F.lit(1)).alias("n_posts")))
        hdst = args.out / "gold" / "city_hourly"
        hourly.write.mode("overwrite").parquet(str(hdst))
        print(f"city_hourly: {spark.read.parquet(str(hdst)).count():,} rows -> {hdst}")

    dst = args.out / "gold" / "city_daily"
    daily.write.mode("overwrite").parquet(str(dst))
    res = spark.read.parquet(str(dst))
    print(f"city_daily: {res.count():,} rows -> {dst}")
    if args.csv:
        csv_dir = args.out / "gold" / "city_daily_csv"
        res.coalesce(1).write.mode("overwrite").option("header", True).csv(str(csv_dir))
        print(f"csv written to {csv_dir}")
    spark.stop()


if __name__ == "__main__":
    main()