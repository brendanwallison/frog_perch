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
    """
    Compute ell_i = log( f_call(p_i) / f_bg(p_i) )
    for an array of detector scores p in (0,1).
    """

    # Evaluate log densities (numerically stable)
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)

    # Log odds ratio
    return log_f_call - log_f_bg


def prepare_stan_data(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
) -> dict:
    """
    Convert a tidy detector DataFrame into a Stan data dictionary
    for the call-intensity model, including precomputed ell_i.

    Removes any rows containing NaN in any required column.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    # Work on a copy
    df = df.copy()

    # Ensure correct dtypes
    df["day_index"] = df["day_index"].astype(int)
    df["time_of_day_hours"] = df["time_of_day_hours"].astype(float)
    df["prob"] = df["prob"].astype(float)

    # ------------------------------------------------------------
    # Remove rows with NaN
    # ------------------------------------------------------------
    before = len(df)
    df = df.dropna(how="any")
    after = len(df)
    removed = before - after

    if removed > 0:
        print(f"[prepare_stan_data] Removed {removed} rows containing NaN values.")

    # ------------------------------------------------------------
    # Precompute ell_i = log f_call(p_i) - log f_bg(p_i)
    # ------------------------------------------------------------
    p_array = df["prob"].to_numpy()
    ell_array = compute_log_odds(
        p_array,
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    # ------------------------------------------------------------
    # Build Stan data dictionary
    # ------------------------------------------------------------
    stan_data = {
        "N": len(df),
        "D": int(df["day_index"].max()),
        "day": df["day_index"].tolist(),
        "t": df["time_of_day_hours"].tolist(),
        "p": df["prob"].tolist(),
        "ell": ell_array.tolist(),   
    }

    return stan_data