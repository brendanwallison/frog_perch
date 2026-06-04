from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import beta
from patsy import dmatrix  #

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


def prepare_stan_data_spline(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    n_ell_bins: int = 10,
    K_season: int = 15,  # New param
    K_diel: int = 15,    # New param
) -> dict:

    df = df.copy()

    # 1. Compute ell_i
    df["ell"] = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    # 2. Compute Time Indices
    # Minute of day (0-1439)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)
    
    # Day of year (1-365)
    # Ensure date is datetime to access .dt accessor
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)

    # 3. Create ell quantile bins
    df["ell_bin"] = pd.qcut(
        df["ell"],
        q=n_ell_bins,
        labels=False,
        duplicates="drop",
    )
    df = df.dropna(subset=["ell_bin"])
    df["ell_bin"] = df["ell_bin"].astype(int)

    # ------------------------------------------------------------
    # 4. Generate Spline Design Matrices (X matrices)
    # ------------------------------------------------------------
    # We only generate spline rows for days/minutes that actually exist in data.
    
    obs_days = np.sort(df["day_of_year"].unique())
    obs_mins = np.sort(df["minute_of_day"].unique())

    # Generate Cyclic Cubic Splines using Patsy
    # df=K controls the complexity (number of columns)
    # bounds set the cycle period (365 days, 1440 minutes)
    X_season = dmatrix(
        f"cc(x, df={K_season}, lower_bound=1, upper_bound=365) - 1", 
        {"x": obs_days}, 
        return_type='dataframe'
    ).values

    X_diel = dmatrix(
        f"cc(x, df={K_diel}, lower_bound=0, upper_bound=1440) - 1", 
        {"x": obs_mins}, 
        return_type='dataframe'
    ).values

    # 5. Create Index Maps (Value -> Matrix Row Index 1..N)
    # Stan uses 1-based indexing
    day_map = {val: i + 1 for i, val in enumerate(obs_days)}
    min_map = {val: i + 1 for i, val in enumerate(obs_mins)}

    # Map the dataframe columns to these new indices
    df["day_idx"] = df["day_of_year"].map(day_map)
    df["diel_idx"] = df["minute_of_day"].map(min_map)

    # ------------------------------------------------------------
    # 6. Group into Bins for Likelihood
    # ------------------------------------------------------------
    # We group by the MATRIX INDICES (day_idx, diel_idx) now, not raw times
    grouped = df.groupby(["day_idx", "diel_idx", "ell_bin"])

    B = len(grouped)
    
    day_idx_vec = []
    diel_idx_vec = []
    ell_bin_vals = []
    n_bin = []

    for (d_idx, m_idx, ebin), g in grouped:
        day_idx_vec.append(int(d_idx))
        diel_idx_vec.append(int(m_idx))
        ell_bin_vals.append(float(g["ell"].mean()))
        n_bin.append(int(len(g)))

    # ------------------------------------------------------------
    # 7. Return Final Data
    # ------------------------------------------------------------
    print(f"[prepare_stan_data] Generated Splines: Season (D_obs={len(obs_days)}, K={X_season.shape[1]}), Diel (M_obs={len(obs_mins)}, K={X_diel.shape[1]})")

    return {
        "B": B,
        "M_obs": len(obs_mins),
        "D_obs": len(obs_days),
        
        # Spline Config
        "K_season": X_season.shape[1],
        "K_diel": X_diel.shape[1],
        "X_season": X_season,
        "X_diel": X_diel,

        # Data Bins
        "day_idx": day_idx_vec,   # Maps bin -> row in X_season
        "diel_idx": diel_idx_vec, # Maps bin -> row in X_diel
        "ell_bin": ell_bin_vals,
        "n_bin": n_bin,
    }


def prepare_stan_data(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    use_binning: bool = False,
    K_season: int = 15,
    K_diel: int = 15,
) -> dict:

    if use_binning:
        print("[prepare_stan_data] Using SPLINE/BINNED mode.")
        return prepare_stan_data_spline(
            df,
            a_call=a_call,
            b_call=b_call,
            a_bg=a_bg,
            b_bg=b_bg,
            K_season=K_season,
            K_diel=K_diel,
        )
    else:
        # Warning: The slice mode is NOT updated for the spline model.
        # If you need slice mode, you must update it to output X_season/X_diel 
        # where M_obs = N_total.
        raise NotImplementedError("Slice mode is not yet compatible with the Spline Stan model.")