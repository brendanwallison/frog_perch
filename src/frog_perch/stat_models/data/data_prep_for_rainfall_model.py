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
# Main Preprocessing: Hydrological Decay + High-Res Climate Matrix
# ---------------------------------------------------------------------
def prepare_stan_data_hydrological(
    df: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_climate: pd.DataFrame,
    calibration_params: dict, 
    *,
    bin_duration_sec: float = 0.2, # Kept for signature compatibility
    window_length_sec: float = 5.0,
    knot_spacing_diel_min: float = 10.0,
    burn_in_days: int = 14,
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict, dict]:

    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            cached_data = pickle.load(f)
            # Ensure compatibility with older 3-item caches
            if len(cached_data) == 4:
                return cached_data
            else:
                print(" > Old cache format detected. Rebuilding data...")

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
        
        x_interf = row['log_mean_rms_1000_1500'] - x_center
        y_mu_norm = np.clip(row['nn_mu'] / k_max, 0.001, 0.999)
        norm_factor = (y_mu_norm * (1.0 - y_mu_norm)) + 1e-6
        nu_obs = (row['nn_var'] / (k_max**2)) / norm_factor
        
        lik_vec = calculate_likelihood_vector(
            row['nn_mu'], nu_obs, x_interf, calibration_params, k_max
        )
        w_obs_list.append(lik_vec)

    w_obs = np.stack(w_obs_list)

    # --- 3. Chronological Rainfall & Burn-in ---
    audio_exact_times = pd.DatetimeIndex([m["start_time"] for m in window_metadata])
    
    audio_start = audio_exact_times.min().normalize()
    audio_end = audio_exact_times.max().normalize()
    sim_start = audio_start - pd.Timedelta(days=burn_in_days)
    
    full_date_range = pd.date_range(sim_start, audio_end, freq='D')
    
    df_rain = df_rain.copy()
    df_rain['Day_DT'] = pd.to_datetime(df_rain['Day'])
    rain_by_day = df_rain.groupby('Day_DT')['Rain_Day'].max()
    
    # --- VERIFICATION TESTS ---
    rain_start_available = rain_by_day.index.min()
    rain_end_available = rain_by_day.index.max()

    print(f"\n=== Timeline Verification ===")
    print(f" > Simulation Start (inc. burn-in): {sim_start.date()}")
    print(f" > Audio Observations Range:      {audio_start.date()} to {audio_end.date()}")
    print(f" > Rainfall Data Range:           {rain_start_available.date()} to {rain_end_available.date()}")

    if rain_start_available > sim_start:
        missing = (rain_start_available - sim_start).days
        print(f" ! WARNING: Rainfall data starts {missing} days AFTER simulation start.")
        print(f"   Indices 0 to {missing-1} will be zero-filled (Dry Cold Start).")
    elif rain_start_available <= sim_start:
        print(f" > OK: Rainfall covers the entire burn-in period.")

    if rain_end_available < audio_end:
        print(f" ! ERROR: Rainfall data ends before audio observations. This will create NaN in likelihood.")
    
    precip_series = rain_by_day.reindex(full_date_range)
    
    internal_gaps = precip_series.isna().sum()
    if internal_gaps > 0:
        print(f" ! WARNING: Found {internal_gaps} missing days within the rainfall timeline (treated as 0mm).")
    
    precip_series = precip_series.fillna(0)
    
    # --- 4. Alignment & Index Integrity ---
    window_start_times = audio_exact_times.normalize()
    day_idx = (window_start_times - sim_start).days.values

    if np.any(day_idx < 0) or np.any(day_idx >= len(full_date_range)):
        raise IndexError("Mapping error: Window day_idx falls outside simulation timeline.")
    
    first_obs_day = full_date_range[day_idx[0]]
    if first_obs_day != audio_start:
         raise ValueError(f"Alignment Mismatch: Index {day_idx[0]} maps to {first_obs_day}, but audio starts at {audio_start}")
    
    print(f" > Alignment Check: First window correctly maps to Day Index {day_idx[0]} ({first_obs_day.date()})")
    
    # --- 5. High-Resolution Climate Data Processing ---
    print(f"\n=== Climate Data Processing ===")
    df_climate = df_climate.copy()
    
    # Drop summary rows: valid timestamps contain a time component (a colon)
    initial_rows = len(df_climate)
    df_climate = df_climate[df_climate['datetime'].str.contains(':', na=False)]
    dropped = initial_rows - len(df_climate)
    if dropped > 0:
        print(f" > Dropped {dropped} summary rows from climate data.")

    # Now safely parse datetimes, enforcing European/Rest-of-World format
    df_climate['datetime'] = pd.to_datetime(df_climate['datetime'], dayfirst=True)
    df_climate = df_climate.drop_duplicates(subset=['datetime']).sort_values('datetime')

    # Step A: Generate Master Diel Profile for Imputation
    df_climate['time_slot_10m'] = df_climate['datetime'].dt.floor('10min').dt.time
    diel_profile = df_climate.groupby('time_slot_10m')[['temp', 'relative_humidity', 'light']].mean()
    
    all_10min_slots = pd.date_range("00:00", "23:50", freq="10min").time
    diel_profile = diel_profile.reindex(all_10min_slots)
    diel_profile = diel_profile.reset_index(drop=True).interpolate(method='linear').bfill().ffill()
    diel_profile.index = all_10min_slots

    # Step B: Isolate Timeline & Resample (Anchored to window_metadata)
    true_audio_start = audio_exact_times.min().floor('D')
    true_audio_end = audio_exact_times.max().ceil('D')
    ideal_climate_index = pd.date_range(start=true_audio_start, end=true_audio_end, freq='10min')

    df_clim_resampled = df_climate.set_index('datetime').reindex(ideal_climate_index)
    df_clim_resampled['date'] = df_clim_resampled.index.date
    df_clim_resampled['time_slot'] = df_clim_resampled.index.time

    # Step C: Gap Detection and Mean-Filling
    valid_counts = df_clim_resampled.groupby('date')['temp'].count()
    expected_per_day = int(24 * 60 / 10)  
    threshold = 0.5 * expected_per_day
    sparse_days = valid_counts[valid_counts < threshold].index 

    imputed_days_count = len(sparse_days)
    for day in sparse_days:
        mask = df_clim_resampled['date'] == day
        for col in ['temp', 'relative_humidity', 'light']:
            df_clim_resampled.loc[mask, col] = df_clim_resampled.loc[mask, 'time_slot'].map(diel_profile[col]).values
        
    print(f" > Imputed {imputed_days_count} sparse climate days.")
    if imputed_days_count > 0:
        print(f"   Example days: {list(sparse_days[:3])}...")

    # Step D: Interpolate to Exact Audio Frame Resolution
    df_audio_target = pd.DataFrame(index=audio_exact_times)
    
    df_combined = pd.concat([
        df_clim_resampled[['temp', 'relative_humidity', 'light']],
        df_audio_target
    ])
    df_combined = df_combined[~df_combined.index.duplicated(keep='first')].sort_index()
    df_combined = df_combined.interpolate(method='time').bfill().ffill()
    
    df_audio_climate = df_combined.reindex(audio_exact_times)

    # Step E: NaN Safety Guardrail
    if df_audio_climate[['temp','relative_humidity','light']].isna().any().any():
        raise ValueError("Climate processing failed: NaNs remain after interpolation/imputation.")

    # --- 6. Splines, Standardization & Packaging ---
    B_diel, K_diel = build_spline_basis(
        np.array(t_mid_list), 
        step_size=knot_spacing_diel_min / 60.0, 
        hard_bounds=(17.0, 23.0)
    )

    climate_scaling = {}
    scaled_climate_cols = []
    
    for col in ['temp', 'relative_humidity', 'light']:
        vals = df_audio_climate[col].values
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        if std_val == 0: std_val = 1.0
        
        scaled_vals = (vals - mean_val) / std_val
        scaled_climate_cols.append(scaled_vals)
        climate_scaling[col] = {"mean": mean_val, "std": std_val}

    X_climate = np.column_stack(scaled_climate_cols)

    stan_data = {
        "T": len(w_obs),
        "N": int(k_max),
        "w_obs": w_obs,
        "K_diel": K_diel,
        "B_diel": B_diel,
        "precip_daily": precip_series.values,
        "day_idx": day_idx,
        "num_days": len(full_date_range),
        "X_climate": X_climate,
        "K_climate": X_climate.shape[1]
    }

    print(f" > Climate matrix (X_climate) compiled and added to stan_data. Shape: {X_climate.shape}")
    print(f"=============================\n")

    spline_params = {"diel_step_min": knot_spacing_diel_min, "burn_in_days": burn_in_days}
    windows_df = pd.DataFrame(window_metadata)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((stan_data, windows_df, spline_params, climate_scaling), f)

    return stan_data, windows_df, spline_params, climate_scaling