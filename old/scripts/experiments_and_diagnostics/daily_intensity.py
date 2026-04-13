#!/usr/bin/env python3
"""
Aggregate per-recording CSV outputs into daily expected call intensities.

Features:
- Reads all CSVs matching pattern like: P2__20241103_110000_SYNC.csv
- Extracts datetime from filename
- Computes mean expected_count per 5s window -> expected calls per minute
- Optional: filter rows to a given time-of-day window
- Differentiates:
    * days WITH recordings but zero call expectation
    * days WITHOUT recordings
- Produces a daily-level CSV and a plot
"""
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
# local config (primary defaults)
import daily_intensity_config as cfg
from datetime import datetime, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Filename datetime extraction
# ---------------------------------------------------------------------------
FNAME_RE = re.compile(r"([0-9]{8})_([0-9]{6})", re.IGNORECASE)

def extract_start_datetime(path: Path) -> Optional[datetime]:
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    dt = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")
    return dt

# ---------------------------------------------------------------------------
# Time filtering
# ---------------------------------------------------------------------------
def filter_time_window(df: pd.DataFrame, start_dt: datetime,
                       start_hour: int, end_hour: int) -> pd.DataFrame:
    """
    Restrict rows to those whose timestamp falls within [start_hour, end_hour).
    """
    if "start_sec" not in df.columns:
        return df  # nothing to filter

    # Compute per-row timestamps
    df = df.copy()
    df["timestamp"] = start_dt + pd.to_timedelta(df["start_sec"], unit="s")

    mask = (df["timestamp"].dt.hour >= start_hour) & (df["timestamp"].dt.hour < end_hour)
    return df.loc[mask]

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def aggregate_daily(day_to_dfs: dict):
    rows = []
    for date_val, df_list in day_to_dfs.items():
        df_all = pd.concat(df_list, ignore_index=True)

        # Mean expected_count per 5s window
        mean_5s = df_all["expected_count"].mean()
        std_5s  = df_all["expected_count"].std(ddof=1)  # sample SD

        # Convert to calls per minute
        mean_60s = mean_5s * 12
        std_60s  = std_5s * 12

        # SEM = SD / sqrt(n)
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

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_plot(daily, out_path):
    plt.figure(figsize=(10,5))
    daily_sorted = daily.sort_values("date")

    plt.errorbar(
        daily_sorted["date"],
        daily_sorted["expected_calls_per_min"],
        yerr=daily_sorted["sem_calls_per_min"],
        fmt="o-", capsize=4
    )

    plt.xlabel("Date")
    plt.ylabel("Expected Calls per Minute")
    plt.title("Daily Call Intensity (mean ± SEM)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Aggregate frog call CSVs to daily intensities.", add_help=True)
    parser.add_argument("--input_dir")
    parser.add_argument("--out_csv")
    parser.add_argument("--plot")
    parser.add_argument("--start_hour", type=int, default=None,
                        help="Filter start hour (0-23). If not set, no filtering.")
    parser.add_argument("--end_hour", type=int, default=None,
                        help="Filter end hour (0-23). If not set, no filtering.")
    args = parser.parse_args()

    input_dir = args.input_dir or cfg.INPUT_DIR
    out_csv = args.out_csv or cfg.OUT_CSV
    plot_file = args.plot or cfg.PLOT
    start_hour = args.start_hour if args.start_hour is not None else cfg.START_HOUR
    end_hour   = args.end_hour   if args.end_hour   is not None else cfg.END_HOUR

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

        # Apply time filtering if requested
        if start_hour is not None and end_hour is not None:
            df = filter_time_window(df, start_dt, start_hour, end_hour)

        # Skip if no rows remain after filtering
        if df.empty:
            continue

        day_to_dfs.setdefault(start_dt.date(), []).append(df)

    if not day_to_dfs:
        print("No valid CSV files found.")
        return

    daily = aggregate_daily(day_to_dfs)

    all_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    full = pd.DataFrame({"date": all_dates})
    merged = full.merge(daily, on="date", how="left")
    merged["no_recordings"] = merged["num_windows"].isna()

    merged.to_csv(out_csv, index=False)

    plot_df = merged.dropna(subset=["expected_calls_per_min"])
    if len(plot_df) > 0:
        make_plot(plot_df, plot_file)
        print(f"Saved plot: {plot_file}")
    else:
        print("No days with recordings to plot.")

    print(f"Saved daily CSV: {out_csv}")

if __name__ == "__main__":
    main()