import re
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# ------------------------------------------------------------
# Filename parsing
# ------------------------------------------------------------

FNAME_RE = re.compile(r"([0-9]{8})_([0-9]{6})", re.IGNORECASE)

def extract_start_datetime(path: Path) -> datetime | None:
    """Extract start datetime from filenames like P2_20241005_183000_SYNC.csv."""
    m = FNAME_RE.search(path.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")


# ------------------------------------------------------------
# 1. Load and merge all detector CSVs
# ------------------------------------------------------------

def load_detector_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Read all detector CSVs in a directory, attach absolute timestamps,
    and return a tidy DataFrame ready for Stan preprocessing.

    Expected CSV columns:
        time_sec, prob, n_eff, var
    Only time_sec and prob are used.
    """
    rows = []

    for path in sorted(csv_dir.glob("*.csv")):
        start_dt = extract_start_datetime(path)
        if start_dt is None:
            continue  # skip files without valid timestamp

        df = pd.read_csv(path)

        # Keep only needed columns
        df = df[["time_sec", "prob"]].copy()

        # Compute absolute timestamps
        df["datetime"] = df["time_sec"].apply(
            lambda s: start_dt + timedelta(seconds=float(s))
        )

        df["date"] = df["datetime"].dt.date

        # Time-of-day in hours (continuous)
        df["time_of_day_hours"] = (
            df["datetime"].dt.hour
            + df["datetime"].dt.minute / 60
            + df["datetime"].dt.second / 3600
        )

        df["path"] = str(path)

        rows.append(df)

    if not rows:
        raise RuntimeError(f"No valid CSVs found in {csv_dir}")

    out = pd.concat(rows, ignore_index=True)

    # Assign day indices (sorted by date)
    unique_dates = sorted(out["date"].unique())
    date_to_index = {d: i + 1 for i, d in enumerate(unique_dates)}
    out["day_index"] = out["date"].map(date_to_index)

    # Sort by absolute time
    out = out.sort_values("datetime").reset_index(drop=True)

    return out
