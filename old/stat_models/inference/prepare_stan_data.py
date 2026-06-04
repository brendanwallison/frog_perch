from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import beta

REQUIRED_COLUMNS = {
    "prob",
    "day_index",
    "time_of_day_hours",
}

def compute_log_odds(p, a_call, b_call, a_bg, b_bg):
    """Compute ell_i = log f_call(p_i) - log f_bg(p_i)."""
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)
    return log_f_call - log_f_bg


# ---------------------------------------------------------------------
# ✅ Slice-mode data prep (your original logic)
# ---------------------------------------------------------------------
def prepare_stan_data_slice(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
) -> dict:

    df = df.copy()

    df["day_index"] = df["day_index"].astype(int)
    df["time_of_day_hours"] = df["time_of_day_hours"].astype(float)
    df["prob"] = df["prob"].astype(float)

    before = len(df)
    df = df.dropna(how="any")
    after = len(df)
    if before - after > 0:
        print(f"[prepare_stan_data] Removed {before - after} rows containing NaN values.")

    ell_array = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    return {
        "N": len(df),
        "D": int(df["day_index"].max()),
        "day": df["day_index"].tolist(),
        "t": df["time_of_day_hours"].tolist(),
        "p": df["prob"].tolist(),
        "ell": ell_array.tolist(),
    }


def prepare_stan_data_binned(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    n_ell_bins: int = 10,
) -> dict:

    df = df.copy()

    # ------------------------------------------------------------
    # Compute ell_i
    # ------------------------------------------------------------
    df["ell"] = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    # Convert time-of-day to integer minute (TRUE minute-of-day)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)

    # ------------------------------------------------------------
    # Compute TRUE day-of-year (1..365)
    # ------------------------------------------------------------
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    # ------------------------------------------------------------
    # Create ell quantile bins
    # ------------------------------------------------------------
    df["ell_bin"] = pd.qcut(
        df["ell"],
        q=n_ell_bins,
        labels=False,
        duplicates="drop",
    )

    df = df.dropna(subset=["ell_bin"])
    df["ell_bin"] = df["ell_bin"].astype(int)

    # ------------------------------------------------------------
    # Unique (day_of_year, minute_of_day) combos → M_obs
    # ------------------------------------------------------------
    dm_df = df[["day_of_year", "minute_of_day"]].drop_duplicates().reset_index(drop=True)
    dm_df["dm_index"] = np.arange(1, len(dm_df) + 1)

    # Merge dm_index back into df
    df = df.merge(dm_df, on=["day_of_year", "minute_of_day"], how="left")

    # ------------------------------------------------------------
    # Group into bins
    # ------------------------------------------------------------
    grouped = df.groupby(["day_of_year", "dm_index", "minute_of_day", "ell_bin"])

    B = len(grouped)
    M_obs = dm_df["dm_index"].max()

    day_id = []
    dm_index = []
    ell_bin_vals = []
    n_bin = []

    for (doy, dm, minute, ebin), g in grouped:
        day_id.append(int(doy))                 # TRUE seasonal index (1..365)
        dm_index.append(int(dm))               # 1..M_obs
        ell_bin_vals.append(float(g["ell"].mean()))
        n_bin.append(int(len(g)))

    # ------------------------------------------------------------
    # Build t_minutes indexed by dm_index (length M_obs)
    # ------------------------------------------------------------
    t_minutes = [0] * M_obs
    for _, row in dm_df.iterrows():
        idx = int(row["dm_index"]) - 1
        t_minutes[idx] = int(row["minute_of_day"])  # TRUE minute-of-day (0..1439)

    # ------------------------------------------------------------
    # Return final Stan data
    # ------------------------------------------------------------
    return {
        "B": B,
        "M_obs": int(M_obs),

        # TRUE periods
        "D_period": 365,
        "N_diel_period": 1440,

        # Bin-level data
        "day_id": day_id,          # TRUE day-of-year (1..365)
        "dm_index": dm_index,      # 1..M_obs
        "ell_bin": ell_bin_vals,
        "n_bin": n_bin,

        # Minute-of-day for each observed (day, minute) combo
        "t_minutes": t_minutes,    # 0..1439
    }

# ---------------------------------------------------------------------
# ✅ Unified entry point with toggle
# ---------------------------------------------------------------------
def prepare_stan_data(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    use_binning: bool = False,
) -> dict:

    if use_binning:
        print("[prepare_stan_data] Using BINNED mode.")
        return prepare_stan_data_binned(
            df,
            a_call=a_call,
            b_call=b_call,
            a_bg=a_bg,
            b_bg=b_bg,
        )
    else:
        print("[prepare_stan_data] Using SLICE mode.")
        return prepare_stan_data_slice(
            df,
            a_call=a_call,
            b_call=b_call,
            a_bg=a_bg,
            b_bg=b_bg,
        )