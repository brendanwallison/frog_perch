#!/usr/bin/env python3
"""
Visualization and Diagnostics for Stan Call-Intensity Model.

Usage:
    python visualize_results.py --output-dir /path/to/results
"""

import argparse
from pathlib import Path
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import arviz as az
from cmdstanpy import from_csv

# Set simpler style
plt.style.use('ggplot')

def load_stan_fit(output_dir: Path):
    """Finds and loads the most recent Stan CSV files in the directory."""
    # Pattern match for cmdstanpy output naming convention
    # usually: name-YYYYMMDDHHMM-chain-X.csv
    csv_files = sorted(glob.glob(str(output_dir / "*.csv")))
    
    # Filter out the metadata/raw data CSVs we saved from Python
    stan_csvs = [f for f in csv_files if "merged_detector" not in f and "windowed_detector" not in f and "raw_bins" not in f]
    
    if not stan_csvs:
        raise FileNotFoundError(f"No Stan output CSV files found in {output_dir}")
    
    print(f"Loading {len(stan_csvs)} Stan CSV chains...")
    fit = from_csv(stan_csvs)
    return fit

def plot_diagnostics(idata, output_dir: Path):
    """Standard MCMC diagnostics (Trace, AutoCorr, R-hat)."""
    print("Generating MCMC diagnostics...")
    
    # 1. Summary Table
    summary = az.summary(idata, var_names=["~z_season", "~z_diel", "~trend_season", "~trend_diel", "~lambda"])
    print("\n=== Parameter Summary (Top 20) ===")
    print(summary.head(20))
    summary.to_csv(output_dir / "fit_summary.csv")

    # 2. Trace Plots (for Hypers)
    # We filter out the big vectors to keep the plot readable
    var_names = ["beta_0", "phi", "sigma_season", "lengthscale_season", "sigma_diel", "lengthscale_diel"]
    
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_trace.png", dpi=150)
    plt.close()

    # 3. Posterior Densities (Hypers)
    az.plot_posterior(idata, var_names=var_names, point_estimate='median', hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_posterior.png", dpi=150)
    plt.close()

    # 4. Energy Plot (BFMI - Check for efficient exploration)
    az.plot_energy(idata)
    plt.savefig(output_dir / "diagnostic_energy.png", dpi=150)
    plt.close()

def plot_hsgp_components(fit, windows_df, output_dir: Path):
    """Plots the reconstructed seasonal and diel trends."""
    print("Generating HSGP component plots...")
    
    # Extract samples (Draws, Chains, Time) -> flatten to (Draws*Chains, Time)
    # Note: CmdStanPy returns (Draws, Chains, Parameters)
    
    # Helper to get summary stats for a vector variable
    def get_trend_summary(param_name):
        # Shape: (N_draws * N_chains, T)
        samples = fit.stan_variable(param_name) 
        mean = np.mean(samples, axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        return mean, lower, upper

    # --- 1. Seasonal Trend ---
    # We plot this against Day of Year
    s_mean, s_lo, s_hi = get_trend_summary("trend_season")
    
    # Sort by day of year for clean line plotting
    doy = windows_df["day_of_year"].values
    sort_idx = np.argsort(doy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(doy[sort_idx], s_mean[sort_idx], color="tab:blue", label="Mean Trend")
    ax.fill_between(doy[sort_idx], s_lo[sort_idx], s_hi[sort_idx], color="tab:blue", alpha=0.3, label="95% CI")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Log-Intensity Contribution")
    ax.set_title("Recovered Seasonal Trend (Log Scale)")
    ax.legend()
    plt.savefig(output_dir / "hsgp_season_trend.png", dpi=150)
    plt.close()

    # --- 2. Diel Trend ---
    # We plot this against Hour of Day
    d_mean, d_lo, d_hi = get_trend_summary("trend_diel")
    
    tod = windows_df["mid_time_hour"].values
    sort_idx = np.argsort(tod)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tod[sort_idx], d_mean[sort_idx], color="tab:orange", label="Mean Trend")
    ax.fill_between(tod[sort_idx], d_lo[sort_idx], d_hi[sort_idx], color="tab:orange", alpha=0.3, label="95% CI")
    ax.set_xlabel("Hour of Day (0-24)")
    ax.set_ylabel("Log-Intensity Contribution")
    ax.set_title("Recovered Diel Cycle (Log Scale)")
    ax.set_xticks(range(0, 25, 4))
    ax.legend()
    plt.savefig(output_dir / "hsgp_diel_trend.png", dpi=150)
    plt.close()

def plot_total_intensity(fit, windows_df, output_dir: Path):
    """Plots the full estimated call rate (Lambda) over absolute time."""
    print("Generating total intensity timeline...")
    
    # Lambda is the actual rate (exp(intercept + season + diel))
    # Shape: (Draws, T)
    lam_samples = fit.stan_variable("lambda")
    
    l_mean = np.mean(lam_samples, axis=0)
    l_lo = np.percentile(lam_samples, 2.5, axis=0)
    l_hi = np.percentile(lam_samples, 97.5, axis=0)
    
    # Parse dates
    dates = pd.to_datetime(windows_df["start_time"])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot standard timeline
    ax.plot(dates, l_mean, color="black", linewidth=1, label="Mean Rate")
    ax.fill_between(dates, l_lo, l_hi, color="gray", alpha=0.4, label="95% CI")
    
    ax.set_title("Estimated Call Intensity Over Time")
    ax.set_ylabel("Calls per Window")
    ax.set_xlabel("Date")
    
    # Format X-axis dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / "total_intensity_timeline.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize Stan HSGP Results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing CSV results")
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.output_dir}")

    # 1. Load Data
    fit = load_stan_fit(args.output_dir)
    
    meta_path = args.output_dir / "windowed_detector_data.csv"
    if not meta_path.exists():
        print("Warning: 'windowed_detector_data.csv' not found. Skipping time-series plots.")
        windows_df = None
    else:
        windows_df = pd.read_csv(meta_path)
    
    # 2. Convert to ArviZ for Diagnostics
    print("Converting to InferenceData...")
    # dimensions mapping helps ArviZ understand the shapes
    idata = az.from_cmdstanpy(
        posterior=fit,
        log_likelihood=None, # Add 'log_lik' here if you calculated it in generated quantities
        dims={
            "trend_season": ["time"],
            "trend_diel": ["time"],
            "lambda": ["time"]
        }
    )

    # 3. Run Visualizations
    plot_diagnostics(idata, args.output_dir)
    
    if windows_df is not None:
        plot_hsgp_components(fit, windows_df, args.output_dir)
        plot_total_intensity(fit, windows_df, args.output_dir)

    print(f"\n✅ Visualization complete. Check {args.output_dir} for PNGs.")

if __name__ == "__main__":
    main()