from __future__ import annotations
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from patsy import dmatrix

# IMPORT THE SINGLE SOURCE OF TRUTH
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

def build_spline_basis(x, step_size, hard_bounds=None):
    if hard_bounds:
        lb, ub = hard_bounds
    else:
        lb, ub = x.min(), x.max()
    knots = np.arange(lb + step_size, ub, step_size)
    design_matrix = dmatrix(
        "bs(x, knots=knots, degree=3, lower_bound=lb, upper_bound=ub, include_intercept=False) - 1", 
        {"x": x, "knots": knots, "lb": lb, "ub": ub},
        return_type='dataframe'
    )
    return design_matrix.values, design_matrix.shape[1]

# ---------------------------------------------------------------------
# Main Preprocessing: Hydrological Decay Version
# ---------------------------------------------------------------------
def prepare_stan_data_hydrological(
    df: pd.DataFrame,
    df_rain: pd.DataFrame,
    calibration_params: dict, 
    *,
    bin_duration_sec: float = 0.2, # Kept for signature compatibility
    window_length_sec: float = 5.0,
    knot_spacing_diel_min: float = 10.0,
    burn_in_days: int = 14,
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:

    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- 1. Audio Data Prep ---
    df = df.copy().dropna(subset=["nn_mu", "nn_var", "datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df["time_of_day_hours"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0 + df["datetime"].dt.second / 3600.0
    df = df[(df["time_of_day_hours"] >= 17.0) & (df["time_of_day_hours"] <= 23.0)]

    # --- 2. Window Generation (Using Shared Math) ---
    t_mid_list = []
    window_metadata = []
    w_obs_list = []
    
    k_max = calibration_params.get("K_MAX", 16.0)
    x_center = calibration_params.get("x_interf_center_mean", 0.0)

    for idx, row in df.iterrows():
        t_mid = row["time_of_day_hours"]
        t_mid_list.append(t_mid)
        window_metadata.append({
            "start_time": row["datetime"],
            "mid_time_hour": t_mid
        })
        
        # Extract features
        x_interf = row['log_mean_rms_1000_1500'] - x_center
        y_mu_norm = np.clip(row['nn_mu'] / k_max, 0.001, 0.999)
        norm_factor = (y_mu_norm * (1.0 - y_mu_norm)) + 1e-6
        nu_obs = (row['nn_var'] / (k_max**2)) / norm_factor
        
        # Get 17-bin probability vector via imported function
        lik_vec = calculate_likelihood_vector(
            row['nn_mu'], nu_obs, x_interf, calibration_params, k_max
        )
        w_obs_list.append(lik_vec)

    w_obs = np.stack(w_obs_list)

    # --- 3. Chronological Rainfall & Burn-in ---
    audio_start = window_metadata[0]["start_time"].normalize()
    audio_end = window_metadata[-1]["start_time"].normalize()
    sim_start = audio_start - pd.Timedelta(days=burn_in_days)
    
    # Timeline for the latent decay process
    full_date_range = pd.date_range(sim_start, audio_end, freq='D')
    
    # Process Rainfall
    df_rain['Day_DT'] = pd.to_datetime(df_rain['Day'])
    rain_by_day = df_rain.groupby('Day_DT')['Rain_Day'].max()
    
    # --- VERIFICATION TESTS ---
    rain_start_available = rain_by_day.index.min()
    rain_end_available = rain_by_day.index.max()

    print(f"\n=== Timeline Verification ===")
    print(f" > Simulation Start (inc. burn-in): {sim_start.date()}")
    print(f" > Audio Observations Range:      {audio_start.date()} to {audio_end.date()}")
    print(f" > Rainfall Data Range:           {rain_start_available.date()} to {rain_end_available.date()}")

    # Test 1: Coverage Check
    if rain_start_available > sim_start:
        missing = (rain_start_available - sim_start).days
        print(f" ! WARNING: Rainfall data starts {missing} days AFTER simulation start.")
        print(f"   Indices 0 to {missing-1} will be zero-filled (Dry Cold Start).")
    elif rain_start_available <= sim_start:
        print(f" > OK: Rainfall covers the entire burn-in period.")

    if rain_end_available < audio_end:
        print(f" ! ERROR: Rainfall data ends before audio observations. This will create NaN in likelihood.")
    
    # Reindex and perform explicit gap detection
    precip_series = rain_by_day.reindex(full_date_range)
    
    # Test 2: Internal Gap Detection
    internal_gaps = precip_series.isna().sum()
    if internal_gaps > 0:
        gap_days = precip_series[precip_series.isna()].index.tolist()
        print(f" ! WARNING: Found {internal_gaps} missing days within the rainfall timeline.")
        print(f"   These will be treated as 0mm rainfall (Dry Gaps).")
    
    # Final zero-fill
    precip_series = precip_series.fillna(0)
    
    # --- 4. Alignment & Index Integrity ---
    window_start_times = pd.Series([m["start_time"] for m in window_metadata]).dt.normalize()
    day_idx = (window_start_times - sim_start).dt.days.values

    # Test 3: Index Boundary Check
    if np.any(day_idx < 0) or np.any(day_idx >= len(full_date_range)):
        raise IndexError("Mapping error: Window day_idx falls outside simulation timeline.")
    
    # Sample Test: Verify first observation alignment
    first_obs_day = full_date_range[day_idx[0]]
    if first_obs_day != audio_start:
         raise ValueError(f"Alignment Mismatch: Index {day_idx[0]} maps to {first_obs_day}, but audio starts at {audio_start}")
    
    print(f" > Alignment Check: First window correctly maps to Day Index {day_idx[0]} ({first_obs_day.date()})")
    print(f"=============================\n")

    # Diel Spline
    B_diel, K_diel = build_spline_basis(
        np.array(t_mid_list), 
        step_size=knot_spacing_diel_min / 60.0, 
        hard_bounds=(17.0, 23.0)
    )

    stan_data = {
        "T": len(w_obs),
        "N": int(k_max),
        "w_obs": w_obs,
        "K_diel": K_diel,
        "B_diel": B_diel,
        "precip_daily": precip_series.values,
        "day_idx": day_idx,
        "num_days": len(full_date_range)
    }

    spline_params = {"diel_step_min": knot_spacing_diel_min, "burn_in_days": burn_in_days}
    windows_df = pd.DataFrame(window_metadata)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((stan_data, windows_df, spline_params), f)

    return stan_data, windows_df, spline_params