# LINE 1: Strict Python requirement for future imports
from __future__ import annotations

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# IMPORT THE SINGLE SOURCE OF TRUTH
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector


# ---------------------------------------------------------------------
# Main Preprocessing: Hydrological Decay + Phase-Shift Coordinate Prep
# ---------------------------------------------------------------------
def prepare_numpyro_data_hydrological(
    df: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_climate: pd.DataFrame,  
    calibration_params: dict, 
    *,
    knot_spacing_diel_min: float = 30.0,
    burn_in_days: int = 14,
    w_fraction: float = 0.0167,
    output_dir: str | Path | None = None,
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:

    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- 1. Audio Data Prep & Sourcing ---
    target_columns = ["nn_mu", "nn_var", "datetime", "mean_rms_1000_1500", "log_mean_rms_1000_1500"]
    df = df.copy().dropna(subset=[col for col in target_columns if col in df.columns])
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df["time_of_day_hours"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0 + df["datetime"].dt.second / 3600.0
    df = df[(df["time_of_day_hours"] >= 18.0) & (df["time_of_day_hours"] <= 22.0)].reset_index(drop=True)

    if "mean_rms_1000_1500" in df.columns:
        rms_global_mean = df["mean_rms_1000_1500"].mean()
        rms_global_std  = df["mean_rms_1000_1500"].std() + 1e-8
        print(f"🌟 Found acoustic predictor. Z-scoring stats: Mean={rms_global_mean:.6f}, Std={rms_global_std:.6f}")
    else:
        raise KeyError("Target column 'mean_rms_1000_1500' was not found in the detector data frames.")

    # --- 2. Robust Climate Continuous Baseline (Global Normalization) ---
    print("🌡️  Processing continuous climate log files for global normalization...")
    df_clim = df_climate.copy()
    df_clim["datetime"] = pd.to_datetime(df_clim["datetime"], errors="coerce", dayfirst=True)
    df_clim = df_clim.dropna(subset=["datetime"])
    
    clim_features = ["temp", "relative_humidity", "light"]
    for col in clim_features:
        if col in df_clim.columns:
            df_clim[col] = pd.to_numeric(df_clim[col], errors="coerce")
            
    df_clim = df_clim.groupby("datetime")[clim_features].mean().reset_index()
    df_clim = df_clim.sort_values("datetime").reset_index(drop=True)
    df_clim = df_clim.dropna(subset=clim_features, how="all")
    
    scale_metrics = {}
    for col in clim_features:
        scale_metrics[f"{col}_mean"] = df_clim[col].mean()
        scale_metrics[f"{col}_std"]  = df_clim[col].std() + 1e-8

    # --- 3. 1-Minute Grid Resampling and Alignment ---
    print("⏳ Interpolating instantaneous climate variables to audio timestamps...")
    
    cols_to_keep = ["datetime"] + clim_features
    df_clim_sub = df_clim[cols_to_keep].copy()
    df_clim_sub = df_clim_sub.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    
    df_clim_continuous = df_clim_sub.resample('1min').interpolate(method='time')
    
    # --- 3b. Align directly to audio timestamps ---
    audio_timestamps = df["datetime"].drop_duplicates().sort_values()
    combined_index = df_clim_continuous.index.union(audio_timestamps).sort_values()
    
    df_clim_interp = df_clim_continuous.reindex(combined_index).interpolate(method="time")
    df_aligned_climate = df_clim_interp.loc[audio_timestamps].reset_index()

    df_aligned_climate = pd.merge(df[["datetime"]], df_aligned_climate, on="datetime", how="left")
    df_aligned_climate = df_aligned_climate.bfill().ffill()

    # --- 3c. Chronological Rainfall ---
    audio_start = df["datetime"].min().normalize()
    audio_end = df["datetime"].max().normalize()
    sim_start = audio_start - pd.Timedelta(days=burn_in_days)
    full_date_range = pd.date_range(sim_start, audio_end, freq='D')
    
    df_rain['Day_DT'] = pd.to_datetime(df_rain['Day'])
    rain_by_day = df_rain.groupby('Day_DT')['Rain_Day'].max()
    precip_series = rain_by_day.reindex(full_date_range).fillna(0)

    # --- Standardize instantaneous arrays ---
    temp_vec  = (df_aligned_climate["temp"].values - scale_metrics["temp_mean"]) / scale_metrics["temp_std"]
    rh_vec    = (df_aligned_climate["relative_humidity"].values - scale_metrics["relative_humidity_mean"]) / scale_metrics["relative_humidity_std"]
    light_vec = (df_aligned_climate["light"].values - scale_metrics["light_mean"]) / scale_metrics["light_std"] 

    # --- 4. Window Generation & Coordinate Sourcing ---
    t_mid_list = []
    window_metadata = []
    w_obs_list = []
    rms_predictor_list = []
    
    k_max = calibration_params.get("K_MAX", 16.0)
    x_center = calibration_params.get("x_interf_center_mean", 0.0)

    for idx, row in df.iterrows():
        t_mid = row["time_of_day_hours"]
        t_mid_list.append(t_mid)
        window_metadata.append({"start_time": row["datetime"], "mid_time_hour": t_mid})
        
        x_interf = row['log_mean_rms_1000_1500'] - x_center
        y_mu_norm = row['nn_mu'] / k_max 
        y_v_norm = row['nn_var'] / (k_max**2)

        lik_vec = calculate_likelihood_vector(y_mu_norm, y_v_norm, x_interf, calibration_params, k_max)
        w_obs_list.append(lik_vec)
        rms_predictor_list.append((row["mean_rms_1000_1500"] - rms_global_mean) / rms_global_std)

    w_obs = np.stack(w_obs_list)
    rms_vec = np.array(rms_predictor_list, dtype=np.float64)

    # --- 5. Window Phase-Shift Coordinates ---
    window_start_times = pd.Series([m["start_time"] for m in window_metadata]).dt.normalize()
    day_idx = (window_start_times - sim_start).dt.days.values

    doy_raw = window_start_times.dt.dayofyear.values
    doy_smooth = (doy_raw - doy_raw.mean()) / (doy_raw.std() + 1e-8)

    # --- 6. Static Spline Parameter Settings & Dimension Calculation ---
    knots_grid = np.arange(18.0 + (knot_spacing_diel_min/60.0), 22.0, knot_spacing_diel_min/60.0)
    K_diel_static = len(knots_grid) + 4

    numpyro_data = {
        "T": len(w_obs),
        "N": int(k_max),
        "w_obs": w_obs,
        "time_of_day": np.array(t_mid_list, dtype=np.float64), 
        "knots_grid": knots_grid,                              
        "K_diel_static": K_diel_static,                        
        "doy_smooth": doy_smooth.astype(np.float64),           
        "precip_daily": precip_series.values,
        "day_idx": day_idx,
        "num_days": len(full_date_range),
        "w_fraction": float(w_fraction),
        "rms_obs": rms_vec,
        
        # Explicit instantaneous standardized vectors
        "temp": temp_vec.astype(np.float64),
        "rh": rh_vec.astype(np.float64),
        "light": light_vec.astype(np.float64),
    }

    spline_params = {"diel_step_min": knot_spacing_diel_min, "burn_in_days": burn_in_days}
    windows_df = pd.DataFrame(window_metadata)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((numpyro_data, windows_df, spline_params), f)

    return numpyro_data, windows_df, spline_params