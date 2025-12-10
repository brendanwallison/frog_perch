#!/usr/bin/env python3
"""
Aggregate Poisson-binomial call-probability CSVs into minute, half-hour,
hour, and daily summaries.

Each input CSV must contain:
    time_sec, prob, n_eff, var   (var is ignored)

Outputs:
    minute/      → one CSV per day
    half_hour/   → one CSV total
    hour/        → one CSV total
    daily/       → one CSV total

All outputs include expected count and variance.
"""

import argparse
import re
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd

SLICE_DURATION_SEC = 5.0 / 16.0  # 0.3125 s

# ---------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------

FNAME_RE = re.compile(r"([0-9]{8})_([0-9]{6})", re.IGNORECASE)

def extract_start_datetime(path: Path) -> datetime | None:
    """Extract start datetime from filenames like P2_20241005_183000_SYNC.csv."""
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")


# ---------------------------------------------------------------------
# Timestamp logic
# ---------------------------------------------------------------------

def add_timestamps(df: pd.DataFrame, start_dt: datetime) -> pd.DataFrame:
    """
    Add absolute timestamps using the *start* of each slice, not the midpoint.
    time_sec in the CSV is the midpoint of a 5/16-second window.
    """
    df = df.copy()

    if "time_sec" not in df.columns:
        raise ValueError("CSV missing required column: time_sec")

    # Convert midpoint → start of window
    df["slice_start_sec"] = df["time_sec"] - 0.15625  # 5/32 seconds

    # Absolute timestamp of the *start* of each slice
    df["timestamp"] = start_dt + pd.to_timedelta(df["slice_start_sec"], unit="s")

    # Period bins
    df["date"] = df["timestamp"].dt.date
    df["minute"] = df["timestamp"].dt.floor("min")
    df["half_hour"] = df["timestamp"].dt.floor("30min")
    df["hour"] = df["timestamp"].dt.floor("H")

    return df


# ---------------------------------------------------------------------
# Poisson-binomial aggregation helpers
# ---------------------------------------------------------------------

def pb_mean(df):
    return df["prob"].sum()

def pb_var(df):
    return (df["prob"] * (1 - df["prob"])).sum()

def summarize_group(df):
    """Return a dict with expected count, variance, and slice count."""
    return {
        "expected_count": pb_mean(df),
        "variance": pb_var(df),
        "num_slices": len(df),
        "date": df["timestamp"].iloc[0].date(),
        "time": df["timestamp"].iloc[0].time(),
    }

def add_per_minute_rate(df):
    """
    Add per-minute call rate and variance, based on Poisson-binomial totals.
    Assumes each slice has fixed duration SLICE_DURATION_SEC.
    """
    df = df.copy()
    # Calls per slice
    df["calls_per_slice"] = df["expected_count"] / df["num_slices"]
    df["var_per_slice"] = df["variance"] / (df["num_slices"] ** 2)

    # Slices per minute
    slices_per_min = 60.0 / SLICE_DURATION_SEC

    # Calls per minute
    df["rate_per_min"] = df["calls_per_slice"] * slices_per_min
    df["rate_var_per_min"] = df["var_per_slice"] * (slices_per_min ** 2)

    return df
# ---------------------------------------------------------------------
# Aggregation routines
# ---------------------------------------------------------------------

def aggregate_minute(df: pd.DataFrame) -> pd.DataFrame:
    """One-minute summaries for a single recording."""
    rows = []
    for _, g in df.groupby("minute"):
        rows.append(summarize_group(g))
    return pd.DataFrame(rows)


def aggregate_half_hour(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df_all.groupby("half_hour"):
        rows.append(summarize_group(g))
    df = pd.DataFrame(rows)
    return add_per_minute_rate(df)


def aggregate_hour(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, g in df_all.groupby("hour"):
        rows.append(summarize_group(g))
    df = pd.DataFrame(rows)
    return add_per_minute_rate(df)


def aggregate_daily(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for date_val, g in df_all.groupby("date"):
        rows.append({
            "date": date_val,
            "expected_count": pb_mean(g),
            "variance": pb_var(g),
            "num_slices": len(g),
        })
    df = pd.DataFrame(rows)
    return add_per_minute_rate(df)


# ---------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------

def main():
    # ---------------------------------------------------------------
    # 1. Try argparse, but do NOT require arguments.
    # ---------------------------------------------------------------
    input_dir = None
    pattern = None

    try:
        import argparse
        parser = argparse.ArgumentParser(
            description="Aggregate Poisson-binomial call CSVs.",
            add_help=True
        )
        parser.add_argument("--input_dir", help="Directory containing CSVs")
        parser.add_argument("--pattern", help="Glob pattern for input CSVs")
        args = parser.parse_args()

        if args.input_dir:
            input_dir = args.input_dir
        if args.pattern:
            pattern = args.pattern

    except Exception:
        pass

    # ---------------------------------------------------------------
    # 2. Fall back to config file if needed
    # ---------------------------------------------------------------
    if input_dir is None or pattern is None:
        try:
            import aggregate_slice_config as cfg
        except ImportError:
            raise RuntimeError(
                "No argparse arguments provided and no aggregate_slice_config.py found."
            )

        if input_dir is None:
            input_dir = cfg.INPUT_DIR
        if pattern is None:
            pattern = getattr(cfg, "PATTERN", "*.csv")

    # ---------------------------------------------------------------
    # 3. Resolve paths and load CSVs
    # ---------------------------------------------------------------
    in_dir = Path(input_dir)
    csv_paths = sorted(in_dir.glob(pattern))

    if not csv_paths:
        print(f"No CSVs found in {in_dir} matching pattern {pattern}")
        return

    # Output folders
    out_minute = in_dir / "minute"
    out_half_hour = in_dir / "half_hour"
    out_hour = in_dir / "hour"
    out_daily = in_dir / "daily"

    for p in [out_minute, out_half_hour, out_hour, out_daily]:
        p.mkdir(exist_ok=True)

    # Collect all data for multi-file summaries
    all_data = []

    # ---------------------------------------------------------------
    # 4. Load all CSVs and accumulate data
    # ---------------------------------------------------------------
    for path in csv_paths:
        start_dt = extract_start_datetime(path)
        if start_dt is None:
            print(f"Skipping {path.name}: cannot extract datetime")
            continue

        df = pd.read_csv(path)
        if "prob" not in df.columns:
            print(f"Skipping {path.name}: missing prob column")
            continue
        # Filter out ghost rows: n_eff == 0 OR prob missing
        df = df[(df["n_eff"] > 0) & (df["prob"].notna())]

        df = add_timestamps(df, start_dt)
        all_data.append(df)

    if not all_data:
        print("No valid data found.")
        return

    df_all = pd.concat(all_data, ignore_index=True)

    # ---------------------------------------------------------------
    # 5. Minute summaries → ONE CSV PER DAY
    # ---------------------------------------------------------------
    for date_val, g in df_all.groupby("date"):
        minute_df = aggregate_minute(g)
        minute_df.to_csv(out_minute / f"{date_val}_minute.csv", index=False)

    # ---------------------------------------------------------------
    # 6. Half-hour summaries → ONE CSV TOTAL
    # ---------------------------------------------------------------
    half_hour_df = aggregate_half_hour(df_all)
    half_hour_df.to_csv(out_half_hour / "half_hour_summary.csv", index=False)

    # ---------------------------------------------------------------
    # 7. Hour summaries → ONE CSV TOTAL
    # ---------------------------------------------------------------
    hour_df = aggregate_hour(df_all)
    hour_df.to_csv(out_hour / "hour_summary.csv", index=False)

    # ---------------------------------------------------------------
    # 8. Daily summary → ONE CSV TOTAL
    # ---------------------------------------------------------------
    daily = aggregate_daily(df_all)
    daily.to_csv(out_daily / "daily_summary.csv", index=False)

    print("Aggregation complete.")

if __name__ == "__main__":
    main()