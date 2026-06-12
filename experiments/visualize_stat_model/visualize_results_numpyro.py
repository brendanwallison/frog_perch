#!/usr/bin/env python3
"""
Visualization and Diagnostics for NumPyro Call-Intensity Model (v5 Hybrid).
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az
import xarray as xr

# Set simpler style
plt.style.use('ggplot')

def load_inference_data(output_dir: Path) -> az.InferenceData:
    """Loads the NetCDF inference data saved by the NumPyro pipeline."""
    nc_path = output_dir / "inference_data.nc"
    
    if not nc_path.exists():
        raise FileNotFoundError(f"Inference Data not found at: {nc_path}")
    
    print(f"Loading InferenceData from {nc_path}...")
    idata = az.from_netcdf(nc_path)
    return idata

def check_divergences(idata):
    """Explicitly checks and prints divergent transitions."""
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        print("Warning: No divergence information found in InferenceData.")
        return

    divergences = idata.sample_stats.diverging.sum().item()
    n_samples = idata.sample_stats.diverging.size
    percent = (divergences / n_samples) * 100
    
    print(f"\n=== HMC/NUTS Diagnostics ===")
    print(f"Divergent Transitions: {divergences} ({percent:.2f}%)")
    
    if divergences > 0:
        print("  ! WARNING: Divergences detected. This indicates validity issues.")
        print("  ! Try increasing 'target_accept_prob' (e.g. 0.90).")
    else:
        print("  > No divergences detected. (Good)")

def plot_mcmc_health(idata, output_dir: Path):
    """Visualizes ESS, R-hat, and Rank Plots."""
    print("Generating MCMC Health plots (ESS, R-hat, Rank)...")
    
    # Critical parameters for v5 Hybrid Model
    var_names = ["beta_0", "phi", "sigma_season", "sigma_diel", "sigma_day"]

    # 1. Rank Plots
    axes = az.plot_rank(idata, var_names=var_names, kind="vlines",
                        vlines_kwargs={'lw':0, 'alpha':0.4})
    plt.suptitle("Rank Plots (Uniform = Good Mixing)")
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_rank.png", dpi=150)
    plt.close()

    # 2. ESS & R-hat Dashboard
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-hat
    az.plot_ess(idata, kind="evolution", var_names=var_names, ax=axes[0])
    axes[0].set_title("ESS Evolution")
    
    # Autocorrelation
    az.plot_autocorr(idata, var_names=var_names, max_lag=20, ax=axes[1])
    axes[1].set_title("Autocorrelation")
    
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_ess_autocorr.png", dpi=150)
    plt.close()

def plot_diagnostics(idata, output_dir: Path):
    """Standard summaries and traces."""
    print("Generating standard diagnostics...")
    
    all_vars = list(idata.posterior.data_vars)
    
    # Exclude high-dimensional vectors from the summary CSV
    exclude_substrings = ["trend_", "lambda", "z_", "beta_season", "beta_diel", "alpha_day"]
    vars_to_show = [
        v for v in all_vars 
        if not any(sub in v for sub in exclude_substrings)
    ]
    
    # Summary Table
    summary = az.summary(idata, var_names=vars_to_show)
    print("\n=== Parameter Summary ===")
    print(summary)
    summary.to_csv(output_dir / "fit_diagnostics.csv")

    # Trace & Posterior
    az.plot_trace(idata, var_names=vars_to_show)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_trace.png", dpi=150)
    plt.close()

    az.plot_posterior(idata, var_names=vars_to_show, point_estimate='median', hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_posterior.png", dpi=150)
    plt.close()

def plot_day_effects(idata, windows_df, output_dir: Path):
    """Plots the 'Weather' offsets (alpha_day) for each unique day."""
    print("Generating Day Random Effects plot...")
    
    if "alpha_day" not in idata.posterior:
        print("  ! 'alpha_day' not found in posterior. Skipping.")
        return

    # Extract alpha_day stats
    # Shape: (chain, draw, num_days)
    da = idata.posterior["alpha_day"]
    means = da.mean(dim=["chain", "draw"]).values
    hdi_low = da.quantile(0.025, dim=["chain", "draw"]).values
    hdi_high = da.quantile(0.975, dim=["chain", "draw"]).values
    
    num_days = len(means)
    
    # Try to map back to dates
    unique_dates = sorted(pd.to_datetime(windows_df["start_time"]).dt.date.unique())
    
    if len(unique_dates) != num_days:
        print(f"  ! Mismatch: Model has {num_days} day params, metadata found {len(unique_dates)} dates.")
        print("  ! Plotting by Index instead.")
        labels = [f"Day {i}" for i in range(num_days)]
        x_vals = range(num_days)
    else:
        labels = unique_dates
        x_vals = unique_dates

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot 'Weather' offsets
    ax.errorbar(x_vals, means, yerr=[means - hdi_low, hdi_high - means], 
                fmt='o', color='tab:purple', alpha=0.7, ecolor='gray', capsize=3)
    
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    
    if len(unique_dates) == num_days:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        plt.xticks(rotation=45)
    
    ax.set_title("Day-Level 'Weather' Effects (alpha_day)")
    ax.set_ylabel("Log-Rate Deviation (Base e)")
    ax.set_xlabel("Date")
    
    plt.tight_layout()
    plt.savefig(output_dir / "day_random_effects.png", dpi=150)
    plt.close()

def plot_spline_components(idata, windows_df, output_dir: Path):
    """Plots the reconstructed seasonal and diel trends (Log Scale)."""
    print("Generating Spline component plots...")
    
    def get_trend_summary(param_name):
        da = idata.posterior[param_name]
        mean = da.mean(dim=["chain", "draw"]).values
        lower = da.quantile(0.025, dim=["chain", "draw"]).values
        upper = da.quantile(0.975, dim=["chain", "draw"]).values
        return mean, lower, upper

    # --- Season ---
    s_mean, s_lo, s_hi = get_trend_summary("trend_season")
    doy = windows_df["day_of_year"].values
    sort_idx = np.argsort(doy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(doy[sort_idx], s_mean[sort_idx], color="tab:blue", label="Mean Trend")
    ax.fill_between(doy[sort_idx], s_lo[sort_idx], s_hi[sort_idx], color="tab:blue", alpha=0.3, label="95% CI")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Log-Intensity Contribution")
    ax.set_title("Recovered Seasonal Trend (Climate)")
    ax.legend()
    plt.savefig(output_dir / "spline_season_trend.png", dpi=150)
    plt.close()

    # --- Diel ---
    d_mean, d_lo, d_hi = get_trend_summary("trend_diel")
    tod = windows_df["mid_time_hour"].values
    sort_idx = np.argsort(tod)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tod[sort_idx], d_mean[sort_idx], color="tab:orange", label="Mean Trend")
    ax.fill_between(tod[sort_idx], d_lo[sort_idx], d_hi[sort_idx], color="tab:orange", alpha=0.3, label="95% CI")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Log-Intensity Contribution")
    ax.set_title("Recovered Diel Cycle (Nightly Habit)")
    ax.set_xticks(range(17, 24))
    ax.legend()
    plt.savefig(output_dir / "spline_diel_trend.png", dpi=150)
    plt.close()

def plot_total_intensity(idata, windows_df, output_dir: Path):
    """Plots the raw estimated lambda (Calls per Window) over time."""
    print("Generating total intensity timeline...")
    
    da_lambda = idata.posterior["lambda"]
    l_mean = da_lambda.mean(dim=["chain", "draw"]).values
    l_lo = da_lambda.quantile(0.025, dim=["chain", "draw"]).values
    l_hi = da_lambda.quantile(0.975, dim=["chain", "draw"]).values
    
    dates = pd.to_datetime(windows_df["start_time"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, l_mean, color="black", linewidth=1, label="Mean Rate")
    ax.fill_between(dates, l_lo, l_hi, color="gray", alpha=0.4, label="95% CI")
    ax.set_title("Estimated Call Intensity Over Time")
    ax.set_ylabel("Calls per Window")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "total_intensity_timeline.png", dpi=150)
    plt.close()

def plot_absolute_decomposition(idata, windows_df, output_dir: Path):
    """
    NEW: Visualizes the model in the Absolute Scale (Calls per Minute).
    Decomposes into 'Nightly Peak Magnitude' and 'Relative Diel Shape'.
    """
    print("Generating Absolute Scale Decomposition (CPM)...")
    
    # --- 0. Calculate Rate Scalar (Window -> Minute) ---
    # We grab the first window to determine the duration
    t_start = pd.to_datetime(windows_df["start_time"].iloc[0])
    t_end   = pd.to_datetime(windows_df["end_time"].iloc[0])
    
    window_sec = (t_end - t_start).total_seconds()
    cpm_scale = 60.0 / window_sec
    
    print(f"  > Detected Window Duration: {window_sec:.2f}s")
    print(f"  > Scaling Factor (to Calls/Min): {cpm_scale:.2f}x")

    # 1. Extract Posterior Samples
    post = idata.posterior.stack(sample=("chain", "draw"))
    
    # --- Part A: The Diel Shape (0-1) ---
    tod = windows_df["mid_time_hour"].values
    
    # Transpose to (sample, time)
    diel_trend_full = post["trend_diel"].transpose("sample", ...).values 
    
    sort_idx = np.argsort(tod)
    tod_sorted = tod[sort_idx]
    diel_sorted = diel_trend_full[:, sort_idx]
    
    diel_exp = np.exp(diel_sorted)
    diel_max = diel_exp.max(axis=1, keepdims=True)
    diel_norm = diel_exp / diel_max
    
    d_mean = diel_norm.mean(axis=0)
    d_lo = np.quantile(diel_norm, 0.025, axis=0)
    d_hi = np.quantile(diel_norm, 0.975, axis=0)

    # --- Part B: The Nightly Peak Magnitude ---
    beta0 = post["beta_0"].values                                     
    season = post["trend_season"].transpose("sample", ...).values     
    alpha = post["alpha_day"].transpose("sample", ...).values         
    
    unique_dates = sorted(pd.to_datetime(windows_df["start_time"]).dt.date.unique())
    num_days = len(unique_dates)
    season_daily = np.zeros((season.shape[0], num_days))
    
    doy = windows_df["day_of_year"].values
    unique_doy = sorted(list(set(doy)))
    
    for i, d in enumerate(unique_doy):
        idx = np.where(doy == d)[0][0]
        season_daily[:, i] = season[:, idx]

    # Calculate Peak Rate (Per Window)
    log_diel_max = np.log(diel_max) 
    log_peak_rate = (beta0[:, None] + season_daily + alpha + log_diel_max)
    peak_rate_window = np.exp(log_peak_rate)
    
    # SCALE TO PER MINUTE
    peak_rate_cpm = peak_rate_window * cpm_scale
    
    p_mean = peak_rate_cpm.mean(axis=0)
    p_lo = np.quantile(peak_rate_cpm, 0.025, axis=0)
    p_hi = np.quantile(peak_rate_cpm, 0.975, axis=0)
    
    # Calculate Potential (Per Minute)
    potential_window = np.exp(beta0[:, None] + season_daily + log_diel_max)
    potential_cpm = potential_window * cpm_scale
    pot_mean = potential_cpm.mean(axis=0)

    # --- Plotting ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), constrained_layout=True)
    
    # Panel 1: Daily Peak Magnitude
    ax0 = axes[0]
    ax0.plot(unique_dates, p_mean, 'o-', color='tab:purple', label="Est. Peak Rate")
    ax0.fill_between(unique_dates, p_lo, p_hi, color='tab:purple', alpha=0.3)
    
    ax0.plot(unique_dates, pot_mean, '--', color='black', alpha=0.7, label="Seasonal Trend (Avg Weather)")
    
    ax0.set_title(f"Nightly Peak Call Intensity (Scaled to Minute)")
    ax0.set_ylabel("Max Calls per Minute")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax0.legend()
    
    # Panel 2: Normalized Diel Schedule
    ax1 = axes[1]
    ax1.plot(tod_sorted, d_mean, color='tab:orange', linewidth=2)
    ax1.fill_between(tod_sorted, d_lo, d_hi, color='tab:orange', alpha=0.3)
    
    ax1.set_title("Relative Diel Schedule (The 'Shape' of the Night)")
    ax1.set_ylabel("Relative Intensity (0 - 1)")
    ax1.set_xlabel("Hour of Day")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(17, 21)
    
    plt.savefig(output_dir / "absolute_decomposition_cpm.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize NumPyro Spline Results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing .nc and .csv results")
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.output_dir}")

    # 1. Load Data
    idata = load_inference_data(args.output_dir)
    
    meta_path = args.output_dir / "windowed_detector_data.csv"
    if not meta_path.exists():
        print("Warning: 'windowed_detector_data.csv' not found. Skipping time-series plots.")
        windows_df = None
    else:
        windows_df = pd.read_csv(meta_path)

    # 2. Run Diagnostics
    check_divergences(idata)
    plot_mcmc_health(idata, args.output_dir)
    plot_diagnostics(idata, args.output_dir)
    
    # 3. Run Visualizations
    if windows_df is not None:
        plot_spline_components(idata, windows_df, args.output_dir)
        plot_day_effects(idata, windows_df, args.output_dir)
        plot_total_intensity(idata, windows_df, args.output_dir)
        plot_absolute_decomposition(idata, windows_df, args.output_dir) # <--- ADDED HERE

    print(f"\n✅ Visualization complete. Check {args.output_dir} for PNGs.")

if __name__ == "__main__":
    main()