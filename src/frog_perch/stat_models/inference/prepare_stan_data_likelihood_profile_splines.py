from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import beta
from patsy import dmatrix  # <--- NEW DEPENDENCY

# ---------------------------------------------------------------------
# Calibration (Unchanged)
# ---------------------------------------------------------------------
def compute_log_odds(p, a_call, b_call, a_bg, b_bg):
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)
    return log_f_call - log_f_bg

# ---------------------------------------------------------------------
# Vectorized PGF / Poisson-Binomial (Unchanged)
# ---------------------------------------------------------------------
def batch_likelihood_vectors(ell_matrix: np.ndarray) -> np.ndarray:
    n_windows, n_bins = ell_matrix.shape
    w_obs = np.zeros((n_windows, n_bins + 1))
    
    for i in range(n_windows):
        L_log_odds = ell_matrix[i] 
        pgf = np.array([1.0])
        for l in L_log_odds:
            Lambda = np.exp(l)
            pgf = np.convolve(pgf, [1.0, Lambda])
        w_obs[i, :] = pgf / np.sum(pgf)
    
    return w_obs

# ---------------------------------------------------------------------
# NEW: Spline Basis Construction
# ---------------------------------------------------------------------
def build_spline_basis(
    x: np.ndarray, 
    step_size: float, 
    hard_bounds: tuple[float, float] | None = None
) -> tuple[np.ndarray, int]:
    """
    Builds B-Spline Design Matrix using Patsy.
    """
    if hard_bounds:
        lb, ub = hard_bounds
    else:
        lb, ub = x.min(), x.max()

    # Generate internal knots strictly inside the bounds
    knots = np.arange(lb + step_size, ub, step_size)
    
    # Generate B-Spline Matrix (Cubic, No Intercept)
    # We subtract 1 to remove the intercept column provided by dmatrix
    # because Stan handles the global beta_0.
    design_matrix = dmatrix(
        "bs(x, knots=knots, degree=3, lower_bound=lb, upper_bound=ub, include_intercept=False) - 1", 
        {"x": x, "knots": knots, "lb": lb, "ub": ub},
        return_type='dataframe'
    )
    
    return design_matrix.values, design_matrix.shape[1]

# ---------------------------------------------------------------------
# Main Preprocessing (Updated for Splines)
# ---------------------------------------------------------------------
def prepare_stan_data_splines(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    bin_duration_sec: float = 0.2,
    window_length_sec: float = 5.0,
    # REPLACED: M_season/M_diel with knot spacings
    knot_spacing_season_days: float = 3.0,     # Every 3 days
    knot_spacing_diel_min: float = 10.0,       # Every 10 minutes
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:

    # Check for cached file
    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            if isinstance(cached_data, tuple) and len(cached_data) == 3:
                print(f"Loaded cached data from {cache_file}")
                return cached_data
            print("Warning: Cache file format mismatch. Recomputing...")

    df = df.copy()
    df = df.dropna(subset=["prob", "datetime"])

    # Compute ell
    df["ell"] = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call, b_call=b_call, a_bg=a_bg, b_bg=b_bg,
    )

    # Sort and Time Prep
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    # Ensure time_of_day_hours exists
    df["time_of_day_hours"] = df["datetime"].dt.hour + \
                              df["datetime"].dt.minute / 60.0 + \
                              df["datetime"].dt.second / 3600.0

    # -------------------------------------------------------
    # CRITICAL: Filter for Hardcoded Diel Range (17:00 - 23:00)
    # -------------------------------------------------------
    # We drop data outside this range BEFORE windowing.
    # The windowing loop will naturally drop windows that try to 
    # bridge the gap (e.g. 23:00 -> 17:00 next day) due to dt_diffs check.
    initial_len = len(df)
    df = df[(df["time_of_day_hours"] >= 17.0) & (df["time_of_day_hours"] <= 23.0)]
    print(f" > Diel Filter (17:00-23:00): Kept {len(df)}/{initial_len} bins.")

    # -------------------------------------------------------
    # DATA REALITY CHECK (Unchanged)
    # -------------------------------------------------------
    time_deltas = df["datetime"].diff().dt.total_seconds().dropna()
    if len(time_deltas) > 0:
        common_interval = time_deltas.round(4).mode()
        if not common_interval.empty:
            inferred_duration = common_interval[0]
            if abs(inferred_duration - bin_duration_sec) > 0.001:
                print(f"  ! WARNING: Mismatch! Data={inferred_duration}s vs Config={bin_duration_sec}s")

    # Determine bins per window
    bins_per_window = int(window_length_sec / bin_duration_sec)
    if bins_per_window < 1:
        raise ValueError("Window length too short for given bin_duration_sec.")
    
    ell_matrix = []
    t_mid_list = []
    doy_list = []
    window_metadata = []
    
    n_total_chunks = 0
    n_dropped_gaps = 0
    n_dropped_tail = 0

    # Sliding windows
    for start_idx in range(0, len(df), bins_per_window):
        n_total_chunks += 1
        end_idx = start_idx + bins_per_window
        
        if end_idx > len(df):
            n_dropped_tail += 1
            break  

        window_df = df.iloc[start_idx:end_idx]
        
        # Check gaps
        dt_diffs = np.diff(window_df["datetime"].astype("int64") / 1e9)
        if not np.allclose(dt_diffs, bin_duration_sec, atol=0.01):
            n_dropped_gaps += 1
            continue  

        # Keep Window
        ell_matrix.append(window_df["ell"].to_numpy())

        # Middle timestamp (hours)
        t_mid = 0.5 * (window_df["time_of_day_hours"].iloc[0] + window_df["time_of_day_hours"].iloc[-1])
        t_mid_list.append(t_mid)

        # Middle day-of-year
        doy = window_df["datetime"].iloc[len(window_df)//2].dayofyear
        doy_list.append(doy)

        window_metadata.append({
            "window_idx": len(ell_matrix), 
            "start_time": window_df["datetime"].iloc[0],
            "end_time": window_df["datetime"].iloc[-1],
            "mid_time_hour": t_mid,
            "day_of_year": doy,
            "n_bins": len(window_df)
        })
    
    # Stats
    n_kept = len(ell_matrix)
    print(f"  > Windows generated: {n_kept}")
    print(f"  > Dropped (Internal Gaps/Boundaries): {n_dropped_gaps}")

    if n_kept == 0:
        raise RuntimeError("No complete windows found.")

    ell_matrix = np.stack(ell_matrix)
    
    # 1. Likelihood Profiles
    print("  > Computing polynomial likelihood profiles...")
    w_obs = batch_likelihood_vectors(ell_matrix)
    
    # 2. Build Spline Basis Functions (REPLACED HSGP)
    print(f"  > Building Spline Basis...")
    
    t_mid_arr = np.array(t_mid_list)
    doy_arr   = np.array(doy_list)
    
    # Season: Adaptive (Min to Max), step = 3 days
    B_season, K_season = build_spline_basis(
        doy_arr, 
        step_size=knot_spacing_season_days
    )
    
    # Diel: Hardcoded (17 to 23), step = 10 mins
    B_diel, K_diel = build_spline_basis(
        t_mid_arr, 
        step_size=knot_spacing_diel_min / 60.0, # Convert mins to hours
        hard_bounds=(17.0, 23.0)
    )
    
    print(f"    Season Cols: {K_season}, Diel Cols: {K_diel}")

    # 1. Get unique days from the list we already built
    unique_doy = sorted(list(set(doy_list)))
    num_days = len(unique_doy)
    
    # 2. Create a map: Day-of-Year -> Contiguous Index (0..D-1)
    doy_map = {d: i for i, d in enumerate(unique_doy)}
    
    # 3. Create the vector for the model
    day_idx = np.array([doy_map[d] for d in doy_list], dtype=int)
    
    print(f"  > Found {num_days} unique days for hierarchical model.")
    # =======================================================

    stan_data = {
        "T": len(w_obs),
        "N": bins_per_window,
        "w_obs": w_obs,
        
        # Spline Data
        "K_season": K_season,
        "B_season": B_season,
        "K_diel": K_diel,
        "B_diel": B_diel,
        
        # Metadata
        "t_mid": t_mid_list,
        "day_of_year": doy_list,
        
        "day_idx": day_idx,   # The vector [0, 0, 1, 1, 2...]
        "num_days": num_days  # The scalar count (e.g. 30)
    }
    
    # Spline configuration for reference/reproducibility
    spline_params = {
        "season_step": knot_spacing_season_days,
        "diel_step_min": knot_spacing_diel_min,
        "diel_bounds": (17.0, 23.0)
    }

    windows_df = pd.DataFrame(window_metadata)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((stan_data, windows_df, spline_params), f)

    return stan_data, windows_df, spline_params