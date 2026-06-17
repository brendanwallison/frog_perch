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
# Helper: Diagnostic Plotter
# ---------------------------------------------------------------------
def _generate_pre_interpolation_diagnostics(df_audio: pd.DataFrame, df_clim_raw: pd.DataFrame, precip_series: pd.Series, output_dir: Path):
    """Generates both Binary (Presence/Absence) and Continuous (Raw Values) heatmaps before alignment."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    min_date = df_audio["datetime"].min().normalize()
    max_date = df_audio["datetime"].max().normalize()
    dates = pd.date_range(min_date, max_date, freq='D').date
    times = pd.date_range("17:00", "23:00", freq="10min").time
    
    df_a = df_audio.copy()
    df_a["date"] = df_a["datetime"].dt.date
    df_a["time_bin"] = df_a["datetime"].dt.floor("10min").dt.time
    audio_binary_pivot = df_a.pivot_table(index="date", columns="time_bin", values="nn_mu", aggfunc="count")
    
    df_c = df_clim_raw.copy().reset_index()
    df_c["datetime"] = pd.to_datetime(df_c["datetime"], errors="coerce")
    df_c = df_c.dropna(subset=["datetime"])
    df_c["time_of_day"] = df_c["datetime"].dt.hour + df_c["datetime"].dt.minute / 60.0
    df_c = df_c[(df_c["time_of_day"] >= 17.0) & (df_c["time_of_day"] <= 23.0)]
    df_c["date"] = df_c["datetime"].dt.date
    df_c["time_bin"] = df_c["datetime"].dt.floor("10min").dt.time

    # --- PLOT 1: BINARY COVERAGE ---
    print(f"📊 Generating pre-interpolation binary coverage diagnostic...")
    fig1, axes1 = plt.subplots(1, 5, figsize=(22, 10), sharey=True, gridspec_kw={'width_ratios': [3, 3, 3, 3, 1]})
    cmap_pres = mcolors.ListedColormap(['#e0e0e0', '#2ecc71'])
    
    def plot_binary_grid(ax, pivot_data, title):
        full_grid = pd.DataFrame(index=dates, columns=times, data=0)
        if not pivot_data.empty:
            aligned = pivot_data.reindex(index=dates, columns=times).notna().astype(int)
            full_grid.update(aligned)
            
        ax.imshow(full_grid.values, aspect="auto", cmap=cmap_pres, vmin=0, vmax=1, interpolation="none")
        ax.set_title(title, fontsize=12, pad=10)
        y_ticks = np.arange(0, len(dates), max(1, len(dates)//15))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([dates[i].strftime("%b %d") for i in y_ticks])
        x_ticks = np.arange(0, len(times), max(1, len(times)//5))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([times[i].strftime("%H:%M") for i in x_ticks])
        ax.set_xlabel("Time (17:00 - 23:00)")

    plot_binary_grid(axes1[0], audio_binary_pivot, "Audio Recording Coverage")
    for i, col in enumerate(["temp", "relative_humidity", "light"], start=1):
        if col in df_c.columns:
            df_c[col] = pd.to_numeric(df_c[col], errors="coerce")
            c_pivot_bin = df_c.pivot_table(index="date", columns="time_bin", values=col, aggfunc="count")
            plot_binary_grid(axes1[i], c_pivot_bin, f"{col.replace('_', ' ').title()} Coverage")
        else:
            plot_binary_grid(axes1[i], pd.DataFrame(), f"{col.title()} (No Data)")

    rain_grid = pd.DataFrame(index=dates, columns=["Rain"], data=0.0)
    valid_rain = precip_series.to_frame(name="rain")
    valid_rain["date"] = valid_rain.index.date
    valid_rain = valid_rain.set_index("date")["rain"]
    rain_grid["Rain"] = valid_rain.reindex(dates).fillna(0.0).values
    im_rain1 = axes1[4].imshow(rain_grid.values, aspect="auto", cmap="Blues", interpolation="none")
    axes1[4].set_title("Daily Rainfall", fontsize=12, pad=10)
    axes1[4].set_xticks([0])
    axes1[4].set_xticklabels(["24h Total"])
    fig1.colorbar(im_rain1, ax=axes1[4], fraction=0.4, pad=0.1, label="Rainfall (mm)")

    plt.suptitle("Pre-Interpolation Binary Coverage (Presence/Absence)", fontsize=16, y=0.96)
    plt.tight_layout()
    plt.savefig(output_dir / "data_coverage_binary_pre.png", dpi=150)
    plt.close()

    # --- PLOT 2: RAW CONTINUOUS VALUES ---
    print(f"📊 Generating pre-interpolation raw continuous values diagnostic...")
    fig2, axes2 = plt.subplots(1, 5, figsize=(22, 10), sharey=True, gridspec_kw={'width_ratios': [3, 3, 3, 3, 1]})
    
    def plot_continuous_grid(ax, pivot_data, title, cmap):
        ax.set_facecolor('#e0e0e0') # Gray background for NaNs (Missing Data)
        full_grid = pd.DataFrame(index=dates, columns=times, data=np.nan)
        if not pivot_data.empty:
            aligned = pivot_data.reindex(index=dates, columns=times)
            full_grid.update(aligned)
            
        im = ax.imshow(full_grid.values, aspect="auto", cmap=cmap, interpolation="none")
        ax.set_title(title, fontsize=12, pad=10)
        y_ticks = np.arange(0, len(dates), max(1, len(dates)//15))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([dates[i].strftime("%b %d") for i in y_ticks])
        x_ticks = np.arange(0, len(times), max(1, len(times)//5))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([times[i].strftime("%H:%M") for i in x_ticks])
        ax.set_xlabel("Time (17:00 - 23:00)")
        return im

    plot_binary_grid(axes2[0], audio_binary_pivot, "Audio Mask (Reference)")
    
    cmaps = ["coolwarm", "BrBG", "magma"]
    for i, (col, cmap) in enumerate(zip(["temp", "relative_humidity", "light"], cmaps), start=1):
        if col in df_c.columns:
            c_pivot_cont = df_c.pivot_table(index="date", columns="time_bin", values=col, aggfunc="mean")
            im = plot_continuous_grid(axes2[i], c_pivot_cont, f"Raw {col.replace('_', ' ').title()}", cmap)
            fig2.colorbar(im, ax=axes2[i], fraction=0.046, pad=0.04)
        else:
            plot_binary_grid(axes2[i], pd.DataFrame(), f"{col.title()} (No Data)")

    im_rain2 = axes2[4].imshow(rain_grid.values, aspect="auto", cmap="Blues", interpolation="none")
    axes2[4].set_title("Daily Rainfall", fontsize=12, pad=10)
    axes2[4].set_xticks([0])
    axes2[4].set_xticklabels(["24h Total"])
    fig2.colorbar(im_rain2, ax=axes2[4], fraction=0.4, pad=0.1, label="Rainfall (mm)")

    plt.suptitle("Pre-Interpolation Raw Continuous Climate Variables", fontsize=16, y=0.96)
    plt.tight_layout()
    plt.savefig(output_dir / "data_coverage_continuous_pre.png", dpi=150)
    plt.close()

def _generate_decomposition_diagnostics(df_clim_double_centered: pd.DataFrame, output_dir: Path):
    """Generates a 3x3 grid proving the mathematical separation of Macro baselines and Micro shocks."""
    print(f"📊 Generating double-centering decomposition diagnostics...")
    
    # Strictly filter the climate diagnostic to the audio window for visual parity
    df_c = df_clim_double_centered.copy().reset_index()
    df_c["time_of_day"] = df_c["datetime"].dt.hour + df_c["datetime"].dt.minute / 60.0
    df_c = df_c[(df_c["time_of_day"] >= 17.0) & (df_c["time_of_day"] <= 23.0)]
    
    min_date = df_c["datetime"].min().normalize()
    max_date = df_c["datetime"].max().normalize()
    dates = pd.date_range(min_date, max_date, freq='D').date
    times = pd.date_range("17:00", "23:00", freq="10min").time 
    
    df_c["date"] = df_c["datetime"].dt.date
    # Floor to 10min so the 10min hardware cadence natively produces the visual "jailbar" gaps
    df_c["time_bin"] = df_c["datetime"].dt.floor("10min").dt.time 
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16), sharex=True, sharey=True)
    
    features = [
        ("temp", "Temperature", "coolwarm", "coolwarm", "RdBu_r"),
        ("relative_humidity", "Rel Humidity", "BrBG", "BrBG", "BrBG"),
        ("light", "Light", "magma", "magma", "PuOr")
    ]
    
    def plot_panel(ax, pivot_data, title, cmap, is_diverging=False):
        ax.set_facecolor('#e0e0e0') # Gray background for NaNs (Hardware gaps)
        full_grid = pd.DataFrame(index=dates, columns=times, data=np.nan)
        if not pivot_data.empty:
            aligned = pivot_data.reindex(index=dates, columns=times)
            full_grid.update(aligned)
            
        # If it is a shock (Intra), force 0.0 to the exact center of the colormap
        vmin, vmax = None, None
        if is_diverging and not pivot_data.isna().all().all():
            abs_max = np.nanmax(np.abs(full_grid.values))
            vmin, vmax = -abs_max, abs_max
            
        im = ax.imshow(full_grid.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="none")
        ax.set_title(title, fontsize=13, pad=10)
        
        y_ticks = np.arange(0, len(dates), max(1, len(dates)//10))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([dates[i].strftime("%b %d") for i in y_ticks])
        
        x_ticks = np.arange(0, len(times), max(1, len(times)//5))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([times[i].strftime("%H:%M") for i in x_ticks])
        return im

    for row_idx, (col, name, cmap_raw, cmap_inter, cmap_intra) in enumerate(features):
        if col not in df_c.columns:
            for col_idx in range(3):
                axes[row_idx, col_idx].set_title(f"{name} (No Data)", fontsize=13)
            continue
            
        # 1. Raw Data
        pivot_raw = df_c.pivot_table(index="date", columns="time_bin", values=col, aggfunc="mean")
        im0 = plot_panel(axes[row_idx, 0], pivot_raw, f"{name} (Raw Continuous)", cmap_raw)
        fig.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # 2. Inter (Macro Baseline)
        pivot_inter = df_c.pivot_table(index="date", columns="time_bin", values=f"{col}_inter", aggfunc="mean")
        im1 = plot_panel(axes[row_idx, 1], pivot_inter, f"{name} (Macro 24h Baseline)", cmap_inter)
        fig.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # 3. Intra (Micro Shock)
        pivot_intra = df_c.pivot_table(index="date", columns="time_bin", values=f"{col}_intra", aggfunc="mean")
        im2 = plot_panel(axes[row_idx, 2], pivot_intra, f"{name} (Micro Intraday Shock)", cmap_intra, is_diverging=True)
        fig.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04, label="Deviation from Diel Mean")

    axes[2, 0].set_xlabel("Time (17:00 - 23:00)", fontsize=11)
    axes[2, 1].set_xlabel("Time (17:00 - 23:00)", fontsize=11)
    axes[2, 2].set_xlabel("Time (17:00 - 23:00)", fontsize=11)

    plt.suptitle("Double-Centering Decomposition (Raw → Inter → Intra)", fontsize=18, y=0.97)
    plt.tight_layout()
    plt.savefig(output_dir / "data_coverage_decomposition.png", dpi=150)
    plt.close()

def _generate_post_interpolation_diagnostics(df_aligned: pd.DataFrame, precip_series: pd.Series, output_dir: Path):
    """Generates Continuous heatmaps of the finalized, interpolated values locked to the audio grid."""
    print(f"📊 Generating post-interpolation aligned continuous values diagnostic...")
    
    min_date = df_aligned["datetime"].min().normalize()
    max_date = df_aligned["datetime"].max().normalize()
    dates = pd.date_range(min_date, max_date, freq='D').date
    times = pd.date_range("17:00", "23:00", freq="10min").time
    
    df_a = df_aligned.copy()
    df_a["date"] = df_a["datetime"].dt.date
    df_a["time_bin"] = df_a["datetime"].dt.floor("10min").dt.time

    fig, axes = plt.subplots(1, 5, figsize=(22, 10), sharey=True, gridspec_kw={'width_ratios': [3, 3, 3, 3, 1]})
    
    def plot_continuous_grid(ax, pivot_data, title, cmap):
        ax.set_facecolor('#e0e0e0') # Gray background for NaNs (Missing Data)
        full_grid = pd.DataFrame(index=dates, columns=times, data=np.nan)
        if not pivot_data.empty:
            aligned = pivot_data.reindex(index=dates, columns=times)
            full_grid.update(aligned)
            
        im = ax.imshow(full_grid.values, aspect="auto", cmap=cmap, interpolation="none")
        ax.set_title(title, fontsize=12, pad=10)
        y_ticks = np.arange(0, len(dates), max(1, len(dates)//15))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([dates[i].strftime("%b %d") for i in y_ticks])
        x_ticks = np.arange(0, len(times), max(1, len(times)//5))
        ax.set_xticks(x_ticks)
        ax.set_xticklabels([times[i].strftime("%H:%M") for i in x_ticks])
        ax.set_xlabel("Time (17:00 - 23:00)")
        return im

    # Use the presence of any interpolated value to draw the audio mask reference
    audio_binary_pivot = df_a.pivot_table(index="date", columns="time_bin", values="datetime", aggfunc="count")
    full_audio_grid = pd.DataFrame(index=dates, columns=times, data=0)
    full_audio_grid.update(audio_binary_pivot.reindex(index=dates, columns=times).notna().astype(int))
    axes[0].imshow(full_audio_grid.values, aspect="auto", cmap=mcolors.ListedColormap(['#e0e0e0', '#2ecc71']), vmin=0, vmax=1, interpolation="none")
    axes[0].set_title("Aligned Audio Mask (Reference)", fontsize=12, pad=10)
    y_ticks = np.arange(0, len(dates), max(1, len(dates)//15))
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels([dates[i].strftime("%b %d") for i in y_ticks])

    cmaps = ["coolwarm", "BrBG", "magma"]
    for i, (col, cmap) in enumerate(zip(["temp", "relative_humidity", "light"], cmaps), start=1):
        if col in df_a.columns:
            # Average the 5-second ticks down to 5-minute bins for visualization
            c_pivot_cont = df_a.pivot_table(index="date", columns="time_bin", values=col, aggfunc="mean")
            im = plot_continuous_grid(axes[i], c_pivot_cont, f"Aligned {col.replace('_', ' ').title()}", cmap)
            fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
        else:
            axes[i].set_title(f"{col.title()} (No Data)", fontsize=12, pad=10)

    rain_grid = pd.DataFrame(index=dates, columns=["Rain"], data=0.0)
    valid_rain = precip_series.to_frame(name="rain")
    valid_rain["date"] = valid_rain.index.date
    valid_rain = valid_rain.set_index("date")["rain"]
    rain_grid["Rain"] = valid_rain.reindex(dates).fillna(0.0).values
    im_rain = axes[4].imshow(rain_grid.values, aspect="auto", cmap="Blues", interpolation="none")
    axes[4].set_title("Daily Rainfall", fontsize=12, pad=10)
    axes[4].set_xticks([0])
    axes[4].set_xticklabels(["24h Total"])
    fig.colorbar(im_rain, ax=axes[4], fraction=0.4, pad=0.1, label="Rainfall (mm)")

    plt.suptitle("Post-Interpolation Aligned Continuous Climate Variables", fontsize=16, y=0.96)
    plt.tight_layout()
    plt.savefig(output_dir / "data_coverage_continuous_post.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Main Preprocessing: Hydrological Decay + Phase-Shift Coordinate Prep
# ---------------------------------------------------------------------
def prepare_numpyro_data_hydrological(
    df: pd.DataFrame,
    df_rain: pd.DataFrame,
    df_climate: pd.DataFrame,  
    calibration_params: dict, 
    *,
    knot_spacing_diel_min: float = 10.0,
    burn_in_days: int = 14,
    w_fraction: float = 0.0167,
    output_dir: str | Path | None = None,
    cache_file: str | None = None,
) -> tuple[dict, pd.DataFrame, dict]:

    if cache_file and Path(cache_file).exists():
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    # --- 1. Audio Data Prep & Sourcing ---
    target_columns = ["nn_mu", "nn_var", "datetime", "mean_rms_1000_1500"]
    df = df.copy().dropna(subset=[col for col in target_columns if col in df.columns])
    
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    
    df["time_of_day_hours"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60.0 + df["datetime"].dt.second / 3600.0
    df = df[(df["time_of_day_hours"] >= 17.0) & (df["time_of_day_hours"] <= 23.0)].reset_index(drop=True)

    if "mean_rms_1000_1500" in df.columns:
        rms_global_mean = df["mean_rms_1000_1500"].mean()
        rms_global_std  = df["mean_rms_1000_1500"].std() + 1e-8
        print(f"🌟 Found acoustic predictor. Z-scoring stats: Mean={rms_global_mean:.6f}, Std={rms_global_std:.6f}")
    else:
        raise KeyError("Target column 'mean_rms_1000_1500' was not found in the detector data frames.")

    # --- 2. Robust Climate Double-Centering (Macro vs. Meso vs. Micro) ---
    print("🌡️  Processing continuous climate log files for double-centering...")
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
    
    print("⏳ Orthogonalizing climate tracks via empirical time-of-day grouping...")
    df_clim["day_key"] = df_clim["datetime"].dt.normalize()
    df_clim["time_bin"] = df_clim["datetime"].dt.floor("10min").dt.time 
    
    for col in clim_features:
        df_clim[f"{col}_inter"] = df_clim.groupby("day_key")[col].transform("mean")
        macro_resid = df_clim[col] - df_clim[f"{col}_inter"]
        diel_baseline = macro_resid.groupby(df_clim["time_bin"]).transform("mean")
        df_clim[f"{col}_intra"] = macro_resid - diel_baseline

    scale_metrics = {}
    for col in clim_features:
        scale_metrics[f"{col}_inter_mean"] = df_clim[f"{col}_inter"].mean()
        scale_metrics[f"{col}_inter_std"]  = df_clim[f"{col}_inter"].std() + 1e-8
        scale_metrics[f"{col}_intra_std"]  = df_clim[f"{col}_intra"].std() + 1e-8

    # --- 3. Robust Time-Aware Interpolation to 5-Second Audio Grid ---
    print("⏳ Interpolating continuous climate predictors onto 5-second acoustic timeline...")
    
    # KEEP the raw feature columns [col] natively here so they interpolate alongside the splines
    cols_to_keep = ["datetime"] + clim_features + [f"{c}_inter" for c in clim_features] + [f"{c}_intra" for c in clim_features]
    df_clim_sub = df_clim[cols_to_keep].copy()
    df_clim_sub = df_clim_sub.drop_duplicates(subset=["datetime"]).set_index("datetime").sort_index()
    
    audio_timestamps = df["datetime"].drop_duplicates().sort_values()
    combined_index = df_clim_sub.index.union(audio_timestamps).sort_values()
    
    df_clim_interp = df_clim_sub.reindex(combined_index).interpolate(method="time")
    df_aligned_climate = df_clim_interp.loc[audio_timestamps].reset_index()
    
    # Merge audio bounds and safely back/forward fill outermost un-interpolated edges
    df_aligned_climate = pd.merge(df[["datetime"]], df_aligned_climate, on="datetime", how="left")
    df_aligned_climate = df_aligned_climate.bfill().ffill()

    # --- 3b. Chronological Rainfall & Seasonal Continuous Coordinates ---
    # (Moved OUTSIDE the if-block so it always evaluates)
    audio_start = df["datetime"].min().normalize()
    audio_end = df["datetime"].max().normalize()
    sim_start = audio_start - pd.Timedelta(days=burn_in_days)
    full_date_range = pd.date_range(sim_start, audio_end, freq='D')
    
    df_rain['Day_DT'] = pd.to_datetime(df_rain['Day'])
    rain_by_day = df_rain.groupby('Day_DT')['Rain_Day'].max()
    precip_series = rain_by_day.reindex(full_date_range).fillna(0)

    # --- DIAGNOSTIC TRIGGER ---
    if output_dir:
        _generate_pre_interpolation_diagnostics(df, df_climate, precip_series, Path(output_dir))
        _generate_post_interpolation_diagnostics(df_aligned_climate, precip_series, Path(output_dir))
        _generate_decomposition_diagnostics(df_clim, Path(output_dir))

    # Apply global continuous standardization to the aligned slices
    temp_inter_vec  = (df_aligned_climate["temp_inter"].values - scale_metrics["temp_inter_mean"]) / scale_metrics["temp_inter_std"]
    temp_intra_vec  = df_aligned_climate["temp_intra"].values / scale_metrics["temp_intra_std"]
    
    rh_inter_vec    = (df_aligned_climate["relative_humidity_inter"].values - scale_metrics["relative_humidity_inter_mean"]) / scale_metrics["relative_humidity_inter_std"]
    rh_intra_vec    = df_aligned_climate["relative_humidity_intra"].values / scale_metrics["relative_humidity_intra_std"]
    
    light_inter_vec = (df_aligned_climate["light_inter"].values - scale_metrics["light_inter_mean"]) / scale_metrics["light_inter_std"]
    light_intra_vec = df_aligned_climate["light_intra"].values / scale_metrics["light_intra_std"]

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
        y_v = row['nn_var'] / (k_max**2)
        lik_vec = calculate_likelihood_vector(row['nn_mu'], y_v, x_interf, calibration_params, k_max)
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
    knots_grid = np.arange(17.0 + (knot_spacing_diel_min/60.0), 23.0, knot_spacing_diel_min/60.0)
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
        "temp_inter":  temp_inter_vec.astype(np.float64),
        "temp_intra":  temp_intra_vec.astype(np.float64),
        "rh_inter":    rh_inter_vec.astype(np.float64),
        "rh_intra":    rh_intra_vec.astype(np.float64),
        "light_inter": light_inter_vec.astype(np.float64),
        "light_intra": light_intra_vec.astype(np.float64)
    }

    spline_params = {"diel_step_min": knot_spacing_diel_min, "burn_in_days": burn_in_days}
    windows_df = pd.DataFrame(window_metadata)

    if cache_file:
        with open(cache_file, "wb") as f:
            pickle.dump((numpyro_data, windows_df, spline_params), f)

    return numpyro_data, windows_df, spline_params