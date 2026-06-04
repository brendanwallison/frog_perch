from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from scipy.stats import beta
from datetime import timedelta

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------
def compute_log_odds(p, a_call, b_call, a_bg, b_bg):
    log_f_call = beta.logpdf(p, a_call, b_call)
    log_f_bg   = beta.logpdf(p, a_bg, b_bg)
    return log_f_call - log_f_bg

# ---------------------------------------------------------------------
# Vectorized PGF / Poisson-Binomial
# ---------------------------------------------------------------------
def batch_likelihood_vectors(ell_matrix: np.ndarray) -> np.ndarray:
    n_windows, n_bins = ell_matrix.shape
    # Output size is N+1 (counts 0 to N)
    w_obs = np.zeros((n_windows, n_bins + 1))
    
    for i in range(n_windows):
        L_log_odds = ell_matrix[i] 
        
        # Start with polynomial [1] (Probability of count 0 is 100%)
        pgf = np.array([1.0])
        
        for l in L_log_odds:
            # Convolve current PGF with [1, Lambda]
            # This represents: (1 * z^0) + (Lambda * z^1)
            Lambda = np.exp(l)
            pgf = np.convolve(pgf, [1.0, Lambda])
            
        # Normalize
        w_obs[i, :] = pgf / np.sum(pgf)
    
    return w_obs

# ---------------------------------------------------------------------
# HSGP Basis Construction (Python Side)
# ---------------------------------------------------------------------
def build_hsgp_basis(x: np.ndarray, M: int, c: float = 1.5) -> tuple[np.ndarray, dict]:
    """
    Builds the Laplacian Eigenfunctions (Sine basis) for HSGP.
    Args:
        x: Input vector (time/day).
        M: Number of basis frequencies.
        c: Boundary factor (domain expansion).
    Returns:
        phi: Design matrix of shape (N_data, M).
        params: Scaling parameters needed for future predictions.
    """
    # 1. Scale x to [-1, 1] for stability
    x_min, x_max = x.min(), x.max()
    x_center = (x_min + x_max) / 2.0
    x_scale  = (x_max - x_min) / 2.0
    
    # Avoid division by zero if single point
    if x_scale == 0: x_scale = 1.0
    
    x_scaled = (x - x_center) / x_scale
    
    # 2. Define Boundary L
    # The basis functions exist on [-L, L]
    L = c 
    
    # 3. Compute Basis Functions
    # phi_m(x) = 1/sqrt(L) * sin( m * pi * (x + L) / (2*L) )
    m_seq = np.arange(1, M + 1) # [1, 2, ..., M]
    
    # Argument shape: (N, 1) * (1, M) -> (N, M)
    arg = (np.pi * m_seq * (x_scaled[:, None] + L)) / (2 * L)
    phi = (1.0 / np.sqrt(L)) * np.sin(arg)
    
    basis_params = {
        "x_center": x_center,
        "x_scale": x_scale,
        "L": L,
        "c": c,
        "M": M
    }
    
    return phi, basis_params

# ---------------------------------------------------------------------
# Main Preprocessing
# ---------------------------------------------------------------------
def prepare_stan_data_vectorized(
    df: pd.DataFrame,
    *,
    a_call: float,
    b_call: float,
    a_bg: float,
    b_bg: float,
    bin_duration_sec: float = 0.2,
    window_length_sec: float = 5.0,
    M_season: int = 10,
    M_diel: int = 10,
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:

    # Check for cached file
    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            # Expecting a tuple of 3 items now
            if isinstance(cached_data, tuple) and len(cached_data) == 3:
                print(f"Loaded cached data from {cache_file}")
                return cached_data
            print("Warning: Cache file format mismatch. Recomputing...")

    df = df.copy()
    df = df.dropna(subset=["prob", "datetime"])

    # Compute ell
    df["ell"] = compute_log_odds(
        df["prob"].to_numpy(),
        a_call=a_call,
        b_call=b_call,
        a_bg=a_bg,
        b_bg=b_bg,
    )

    # Sort by absolute time and ensure datetime format
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # -------------------------------------------------------
    # DATA REALITY CHECK
    # -------------------------------------------------------
    # Calculate time steps between consecutive rows
    time_deltas = df["datetime"].diff().dt.total_seconds().dropna()
    
    # Find the most common interval (mode) rounded to 4 decimals
    if len(time_deltas) > 0:
        common_interval = time_deltas.round(4).mode()
        if not common_interval.empty:
            inferred_duration = common_interval[0]
            
            print(f"  > Data Check: Most common row interval is {inferred_duration}s")
            print(f"  > User Setting: bin_duration_sec = {bin_duration_sec}s")
            
            if abs(inferred_duration - bin_duration_sec) > 0.001:
                print(f"  ! WARNING: Significant mismatch detected!")
                print( "    The data does not match your bin_duration setting.")
                print( "    This will likely cause the windowing loop to drop ALL data.")
    # -------------------------------------------------------

    # Determine bins per window
    bins_per_window = int(window_length_sec / bin_duration_sec)
    if bins_per_window < 1:
        raise ValueError("Window length too short for given bin_duration_sec.")
    
    print(f"  > Bins per window: {bins_per_window}")

    ell_matrix = []
    t_mid_list = []
    doy_list = []
    window_metadata = []
    
    # Counters for statistics
    n_total_chunks = 0
    n_dropped_gaps = 0
    n_dropped_tail = 0

    # Sliding windows over consecutive bins, drop any incomplete windows
    for start_idx in range(0, len(df), bins_per_window):
        n_total_chunks += 1
        end_idx = start_idx + bins_per_window
        
        # Check 1: Incomplete window at the very end of the file
        if end_idx > len(df):
            n_dropped_tail += 1
            break  

        window_df = df.iloc[start_idx:end_idx]
        
        # Check 2: Gaps within the window (e.g. recording stopped)
        dt_diffs = np.diff(window_df["datetime"].astype("int64") / 1e9)  # seconds
        if not np.allclose(dt_diffs, bin_duration_sec, atol=0.01):
            n_dropped_gaps += 1
            continue  

        # --- Data for Stan ---
        ell_vec = window_df["ell"].to_numpy()
        ell_matrix.append(ell_vec)

        # Middle timestamp (hours)
        t_start = window_df["time_of_day_hours"].iloc[0]
        t_end   = window_df["time_of_day_hours"].iloc[-1]
        t_mid = 0.5 * (t_start + t_end)
        t_mid_list.append(t_mid)

        # Middle day-of-year
        doy_start = pd.to_datetime(window_df["datetime"].iloc[0]).dayofyear
        doy_end   = pd.to_datetime(window_df["datetime"].iloc[-1]).dayofyear
        doy = int(round(0.5 * (doy_start + doy_end)))
        doy_list.append(doy)

        # --- Metadata for DataFrame ---
        window_metadata.append({
            "window_idx": len(ell_matrix), # 1-based index (matches Stan loop)
            "start_time": window_df["datetime"].iloc[0],
            "end_time": window_df["datetime"].iloc[-1],
            "mid_time_hour": t_mid,
            "day_of_year": doy,
            "n_bins": len(window_df)
        })
    
    # Print Data Loss Statistics
    n_kept = len(ell_matrix)
    print(f"  > Total chunks attempted: {n_total_chunks}")
    print(f"  > Windows kept: {n_kept}")
    print(f"  > Dropped (Internal Gaps): {n_dropped_gaps}")
    print(f"  > Dropped (End Tail): {n_dropped_tail}")

    if n_kept == 0:
        raise RuntimeError("No complete windows found. Check bin duration and gaps.")

    ell_matrix = np.stack(ell_matrix)
    
    # 1. Compute Aggregated Likelihood Profiles
    print("  > Computing polynomial likelihood profiles...")
    w_obs = batch_likelihood_vectors(ell_matrix)
    
    # 2. Compute HSGP Basis Functions
    print(f"  > Building HSGP Basis Functions (Season M={M_season}, Diel M={M_diel})...")
    t_mid_arr = np.array(t_mid_list)
    doy_arr   = np.array(doy_list)
    
    X_season, params_season = build_hsgp_basis(doy_arr, M=M_season, c=1.5)
    X_diel,   params_diel   = build_hsgp_basis(t_mid_arr, M=M_diel, c=1.2)
    
    hsgp_params = {
        "season": params_season,
        "diel": params_diel
    }

    stan_data = {
        "T": len(w_obs),
        "N": bins_per_window,
        "w_obs": w_obs,
        
        # Covariates (Reference only)
        "t_mid": t_mid_list,
        "day_of_year": doy_list,
        
        # HSGP Matrices & Sizes
        "M_season": M_season,
        "M_diel": M_diel,
        "X_season": X_season,
        "X_diel": X_diel,
        "L_season": hsgp_params["season"]["L"],
        "L_diel": hsgp_params["diel"]["L"],
        
        # Configs
        "window_length_sec": window_length_sec,
        "bin_duration_sec": bin_duration_sec,
    }

    # 3. Organize Outputs
    windows_df = pd.DataFrame(window_metadata)

    # Save cache if requested (saving triplet)
    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((stan_data, windows_df, hsgp_params), f)

    return stan_data, windows_df, hsgp_params