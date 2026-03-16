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
    and return a tidy DataFrame ready for calibration and Stan preprocessing.
    
    Preserves all extracted features (nn_mu, nn_var, covariates, etc.).
    """
    rows = []
    csv_paths = list(Path(csv_dir).glob("*.csv"))

    if not csv_paths:
        raise RuntimeError(f"No CSVs found in {csv_dir}")

    for path in sorted(csv_paths):
        start_dt = extract_start_datetime(path)
        if start_dt is None:
            continue  # skip files without valid timestamp

        df = pd.read_csv(path)

        if "time_sec" not in df.columns:
            print(f"[WARN] Skipping {path.name}: missing 'time_sec' column.")
            continue

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

        df["source_file"] = path.name
        rows.append(df)

    if not rows:
        raise RuntimeError(f"No valid data could be loaded from CSVs in {csv_dir}")

    out = pd.concat(rows, ignore_index=True)

    # Assign day indices (sorted by date)
    unique_dates = sorted(out["date"].unique())
    date_to_index = {d: i + 1 for i, d in enumerate(unique_dates)}
    out["day_index"] = out["date"].map(date_to_index)

    # Sort by absolute time
    out = out.sort_values("datetime").reset_index(drop=True)

    return out