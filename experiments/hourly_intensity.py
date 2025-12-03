#!/usr/bin/env python3
"""
Aggregate per-recording CSV outputs into daily/hourly expected call intensities.

Features:
- Reads all CSVs matching pattern like: P2_20241005_183000_SYNC.csv
- Extracts full datetime from filename
- Computes per-row timestamps using start_sec offsets
- Aggregates expected_count into daily/hourly means, SD, SEM
- Produces two CSVs:
    * daily averages
    * hourly averages (only hours with recordings)
- Produces a plot with one line per hour (mean ± SEM)
"""

import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.cm as cm
import matplotlib.colors as mcolors


# local config (primary defaults)
import hourly_intensity_config as cfg

FNAME_RE = re.compile(r"([0-9]{8})_([0-9]{6})", re.IGNORECASE)

def extract_start_datetime(path: Path) -> datetime | None:
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")

def add_timestamps(df: pd.DataFrame, start_dt: datetime) -> pd.DataFrame:
    """Add absolute timestamps to each row using start_sec offsets."""
    if "start_sec" not in df.columns:
        return df
    df = df.copy()
    df["timestamp"] = start_dt + pd.to_timedelta(df["start_sec"], unit="s")
    return df

def aggregate_daily(day_to_dfs: dict) -> pd.DataFrame:
    rows = []
    for date_val, df_list in day_to_dfs.items():
        df_all = pd.concat(df_list, ignore_index=True)
        mean_5s = df_all["expected_count"].mean()
        std_5s  = df_all["expected_count"].std(ddof=1)
        mean_60s = mean_5s * 12
        std_60s  = std_5s * 12
        n = len(df_all)
        sem_60s = std_60s / (n**0.5) if n > 0 else float("nan")
        rows.append({
            "date": date_val,
            "expected_calls_per_min": mean_60s,
            "std_calls_per_min": std_60s,
            "sem_calls_per_min": sem_60s,
            "num_windows": n,
            "num_recordings": len(df_list),
        })
    return pd.DataFrame(rows)

def aggregate_hourly(day_to_dfs: dict) -> pd.DataFrame:
    rows = []
    for date_val, df_list in day_to_dfs.items():
        df_all = pd.concat(df_list, ignore_index=True)
        if "timestamp" not in df_all.columns:
            continue
        df_all["hour"] = df_all["timestamp"].dt.hour
        for hour, df_hour in df_all.groupby("hour"):
            mean_5s = df_hour["expected_count"].mean()
            std_5s  = df_hour["expected_count"].std(ddof=1)
            mean_60s = mean_5s * 12
            std_60s  = std_5s * 12
            n = len(df_hour)
            sem_60s = std_60s / (n**0.5) if n > 0 else float("nan")
            rows.append({
                "date": date_val,
                "hour": hour,
                "expected_calls_per_min": mean_60s,
                "std_calls_per_min": std_60s,
                "sem_calls_per_min": sem_60s,
                "num_windows": n,
            })
    return pd.DataFrame(rows)

def make_plot_hourly(hourly: pd.DataFrame, out_path: str):
    plt.figure(figsize=(10,5))
    hourly_sorted = hourly.sort_values(["hour","date"])

    # Build a colormap scaled to the hours present
    hours = sorted(hourly_sorted["hour"].unique())
    cmap = plt.get_cmap("viridis", len(hours))  # sample evenly spaced colors

    for idx, hour in enumerate(hours):
        df_hour = hourly_sorted[hourly_sorted["hour"] == hour]
        color = cmap(idx)  # evenly spaced distinct colors
        plt.errorbar(
            df_hour["date"],
            df_hour["expected_calls_per_min"],
            yerr=df_hour["sem_calls_per_min"],
            fmt="-o",
            capsize=1,
            markersize=2,
            linewidth=1,
            color=color,
            label=f"{hour:02d}:00"
        )


    plt.xlabel("Date")
    plt.ylabel("Expected Calls per Minute")
    plt.title("Daily Call Intensity by Hour (mean ± SEM)")
    plt.xticks(rotation=45)
    plt.legend(title="Hour of Day", ncol=2, fontsize="small")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)  # higher resolution
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Aggregate frog call CSVs to daily/hourly intensities.", add_help=True)
    parser.add_argument("--input_dir", help="Directory containing *_SYNC.csv files")
    parser.add_argument("--out_csv", help="Base name for output CSVs")
    parser.add_argument("--plot", help="Output plot filename (PNG)")
    args = parser.parse_args()

    # fallback to config
    input_dir = args.input_dir or cfg.INPUT_DIR
    out_csv   = args.out_csv   or cfg.OUT_CSV
    plot_file = args.plot      or cfg.PLOT

    in_dir = Path(input_dir)
    csvs = sorted(in_dir.glob("*_SYNC.csv"))

    day_to_dfs = {}
    for path in csvs:
        start_dt = extract_start_datetime(path)
        if start_dt is None:
            continue
        df = pd.read_csv(path)
        if "expected_count" not in df.columns:
            continue
        df = add_timestamps(df, start_dt)
        if df.empty:
            continue
        day_to_dfs.setdefault(start_dt.date(), []).append(df)

    if not day_to_dfs:
        print("No valid CSV files found.")
        return

    daily = aggregate_daily(day_to_dfs)
    hourly = aggregate_hourly(day_to_dfs)

    daily.to_csv(out_csv.replace(".csv","_daily.csv"), index=False)
    hourly.to_csv(out_csv.replace(".csv","_hourly.csv"), index=False)

    if not hourly.empty:
        make_plot_hourly(hourly, plot_file)
        print(f"Saved hourly plot: {plot_file}")
    else:
        print("No hourly data to plot.")

    print(f"Saved daily CSV: {out_csv.replace('.csv','_daily.csv')}")
    print(f"Saved hourly CSV: {out_csv.replace('.csv','_hourly.csv')}")

if __name__ == "__main__":
    main()