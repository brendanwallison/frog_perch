from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import beta
from patsy import dmatrix

# --- HARD CODED WINDOW (Must match Analysis!) ---
SEASON_START = 240
SEASON_END   = 320
HOUR_START   = 17.0
HOUR_END     = 21.0

def compute_log_odds(p, a_call, b_call, a_bg, b_bg):
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)
    return log_f_call - log_f_bg

def get_linear_bspline_matrix(x_vals, min_val, max_val, n_dofs, degree=3):
    """
    Generates B-Spline matrix with FIXED linear knots.
    """
    # Calculate inner knots to create exactly n_dofs columns
    # We subtract 1 because we use include_intercept=True
    n_inner = n_dofs - degree - 1
    
    if n_inner < 0:
        # Fallback for small K
        return dmatrix(
             f"bs(x, df={n_dofs}, degree={degree}, include_intercept=True, lower_bound={min_val}, upper_bound={max_val}) - 1",
             {"x": x_vals}, return_type='dataframe'
        ).values
        
    knots = np.linspace(min_val, max_val, n_inner + 2)[1:-1]
    
    return dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True, lower_bound=min_val, upper_bound=max_val) - 1",
        {"x": x_vals, "knots": knots, "degree": degree, "min_val": min_val, "max_val": max_val},
        return_type='dataframe'
    ).values

def prepare_stan_data_windowed(
    df: pd.DataFrame,
    *,
    a_call: float, b_call: float, a_bg: float, b_bg: float,
    n_ell_bins: int = 10,
    K_season: int = 20, 
    K_diel: int = 20,
) -> dict:
    
    df = df.copy()
    
    # 1. Filter Window
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)

    mask_season = df["day_of_year"].between(SEASON_START, SEASON_END)
    mask_diel   = df["time_of_day_hours"].between(HOUR_START, HOUR_END)
    df = df[mask_season & mask_diel].reset_index(drop=True)

    if len(df) == 0: raise ValueError("Data filtering removed all rows!")
    print(f"Data filtered to window. Rows: {len(df)}")

    # 2. Prep Data
    df["ell"] = compute_log_odds(df["prob"].values, a_call, b_call, a_bg, b_bg)
    df["ell_bin"] = pd.qcut(df["ell"], q=n_ell_bins, labels=False, duplicates="drop")
    df = df.dropna(subset=["ell_bin"])
    df["ell_bin"] = df["ell_bin"].astype(int)

    # 3. Generate Splines
    obs_days = np.sort(df["day_of_year"].unique())
    obs_mins = np.sort(df["minute_of_day"].unique())

    # Use robust generator
    X_season = get_linear_bspline_matrix(obs_days, SEASON_START, SEASON_END, K_season)
    X_diel   = get_linear_bspline_matrix(obs_mins, HOUR_START*60, HOUR_END*60, K_diel)

    # 4. Pack
    day_map = {val: i + 1 for i, val in enumerate(obs_days)}
    min_map = {val: i + 1 for i, val in enumerate(obs_mins)}

    df["day_idx"] = df["day_of_year"].map(day_map)
    df["diel_idx"] = df["minute_of_day"].map(min_map)

    grouped = df.groupby(["day_idx", "diel_idx", "ell_bin"])
    B = len(grouped)
    
    day_idx_vec, diel_idx_vec, ell_bin_vals, n_bin = [], [], [], []
    for (d, m, e), g in grouped:
        day_idx_vec.append(int(d))
        diel_idx_vec.append(int(m))
        ell_bin_vals.append(float(g["ell"].mean()))
        n_bin.append(int(len(g)))

    return {
        "B": B, "M_obs": len(obs_mins), "D_obs": len(obs_days),
        "K_season": X_season.shape[1], "K_diel": X_diel.shape[1],
        "X_season": X_season, "X_diel": X_diel,
        "day_idx": day_idx_vec, "diel_idx": diel_idx_vec,
        "ell_bin": ell_bin_vals, "n_bin": n_bin,
    }

def prepare_stan_data(df, **kwargs):
    # Compatibility wrapper
    K_season = kwargs.get("K_season", kwargs.get("M_season", 20))
    K_diel   = kwargs.get("K_diel", kwargs.get("M_diel", 20))
    return prepare_stan_data_windowed(
        df, a_call=kwargs["a_call"], b_call=kwargs["b_call"],
        a_bg=kwargs["a_bg"], b_bg=kwargs["b_bg"],
        K_season=K_season, K_diel=K_diel
    )