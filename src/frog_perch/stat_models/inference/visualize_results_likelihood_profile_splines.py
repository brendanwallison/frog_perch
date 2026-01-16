#!/usr/bin/env python3
"""
Visualization and Diagnostics for Stan Call-Intensity Model (P-Splines).
Updated with Rank Plots, ESS Analysis, and Divergence Checks.
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

plt.style.use('ggplot')

def load_stan_fit(output_dir: Path):
    """Finds and loads the most recent Stan CSV files."""
    csv_files = sorted(glob.glob(str(output_dir / "*.csv")))
    stan_csvs = [
        f for f in csv_files 
        if "merged_detector" not in f 
        and "windowed_detector" not in f 
        and "raw_bins" not in f
    ]
    
    if not stan_csvs:
        raise FileNotFoundError(f"No Stan output CSV files found in {output_dir}")
    
    print(f"Loading {len(stan_csvs)} Stan CSV chains...")
    fit = from_csv(stan_csvs)
    return fit

def check_divergences(idata):
    """Explicitly checks and prints divergent transitions."""
    divergences = idata.sample_stats.diverging.sum().item()
    n_samples = idata.sample_stats.diverging.size
    percent = (divergences / n_samples) * 100
    
    print(f"\n=== Hamiltonian Monte Carlo Diagnostics ===")
    print(f"Divergent Transitions: {divergences} ({percent:.2f}%)")
    
    if divergences > 0:
        print("  ! WARNING: Divergences detected. This indicates validity issues.")
        print("  ! Try increasing 'adapt_delta' (e.g., 0.90, 0.95) or simplifying the model.")
    else:
        print("  > No divergences detected. (Good)")

def plot_mcmc_health(idata, output_dir: Path):
    """Visualizes ESS, R-hat, and Rank Plots."""
    print("Generating MCMC Health plots (ESS, R-hat, Rank)...")
    
    # We focus on the main scalar parameters for these plots
    var_names = ["beta_0", "phi", "sigma_season", "sigma_diel"]

    # 1. Rank Plots (The modern "Trace Plot")
    # Ideally, these look like uniform histograms. 
    # If they are slanted or u-shaped, chains are not mixing.
    axes = az.plot_rank(idata, var_names=var_names, kind="vlines",
                        vlines_kwargs={'lw':0, 'alpha':0.4})
    plt.suptitle("Rank Plots (Uniform = Good Mixing)")
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_rank.png", dpi=150)
    plt.close()

    # 2. ESS & R-hat (Visual Summary)
    # This plots the ESS and R-hat for *all* parameters (including the vectors)
    # creating a quick dashboard to see if any specific param is stuck.
    
    # We exclude the raw spline weights (beta/z) to keep the plot from being too dense,
    # focusing on the trends and hypers.
    vars_to_plot = ["~z_season", "~z_diel", "~beta_season", "~beta_diel"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # R-hat (Convergence)
    az.plot_ess(idata, kind="evolution", var_names=var_names, ax=axes[0])
    axes[0].set_title("ESS Evolution (Bulk)")
    
    # Autocorrelation (Mixing)
    az.plot_autocorr(idata, var_names=var_names, max_lag=20, ax=axes[1])
    axes[1].set_title("Autocorrelation")
    
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_ess_autocorr.png", dpi=150)
    plt.close()

def plot_diagnostics(idata, output_dir: Path):
    """Standard summaries and traces."""
    print("Generating standard diagnostics...")
    
    # Summary Table with specific focus on ESS columns
    exclude_vars = [
        "~z_season", "~z_diel", 
        "~beta_season", "~beta_diel", 
        "~trend_season", "~trend_diel", 
        "~lambda"
    ]
    summary = az.summary(idata, var_names=exclude_vars, kind="diagnostics")
    print("\n=== Effective Sample Size & R-hat ===")
    print(summary)
    summary.to_csv(output_dir / "fit_diagnostics.csv")

    # Trace & Posterior
    var_names = ["beta_0", "phi", "sigma_season", "sigma_diel"]
    
    az.plot_trace(idata, var_names=var_names)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_trace.png", dpi=150)
    plt.close()

    az.plot_posterior(idata, var_names=var_names, point_estimate='median', hdi_prob=0.95)
    plt.tight_layout()
    plt.savefig(output_dir / "diagnostic_posterior.png", dpi=150)
    plt.close()

def plot_spline_components(fit, windows_df, output_dir: Path):
    # (Same as before)
    print("Generating Spline component plots...")
    
    def get_trend_summary(param_name):
        samples = fit.stan_variable(param_name) 
        mean = np.mean(samples, axis=0)
        lower = np.percentile(samples, 2.5, axis=0)
        upper = np.percentile(samples, 97.5, axis=0)
        return mean, lower, upper

    # Season
    s_mean, s_lo, s_hi = get_trend_summary("trend_season")
    doy = windows_df["day_of_year"].values
    sort_idx = np.argsort(doy)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(doy[sort_idx], s_mean[sort_idx], color="tab:blue", label="Mean Trend")
    ax.fill_between(doy[sort_idx], s_lo[sort_idx], s_hi[sort_idx], color="tab:blue", alpha=0.3, label="95% CI")
    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Log-Intensity")
    ax.set_title("Recovered Seasonal Trend")
    ax.legend()
    plt.savefig(output_dir / "spline_season_trend.png", dpi=150)
    plt.close()

    # Diel
    d_mean, d_lo, d_hi = get_trend_summary("trend_diel")
    tod = windows_df["mid_time_hour"].values
    sort_idx = np.argsort(tod)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tod[sort_idx], d_mean[sort_idx], color="tab:orange", label="Mean Trend")
    ax.fill_between(tod[sort_idx], d_lo[sort_idx], d_hi[sort_idx], color="tab:orange", alpha=0.3, label="95% CI")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Log-Intensity")
    ax.set_title("Recovered Diel Cycle")
    ax.set_xticks(range(0, 25, 4))
    ax.legend()
    plt.savefig(output_dir / "spline_diel_trend.png", dpi=150)
    plt.close()

def plot_total_intensity(fit, windows_df, output_dir: Path):
    # (Same as before)
    print("Generating total intensity timeline...")
    lam_samples = fit.stan_variable("lambda")
    l_mean = np.mean(lam_samples, axis=0)
    l_lo = np.percentile(lam_samples, 2.5, axis=0)
    l_hi = np.percentile(lam_samples, 97.5, axis=0)
    
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

def main():
    parser = argparse.ArgumentParser(description="Visualize Stan Spline Results")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory containing CSV results")
    args = parser.parse_args()

    if not args.output_dir.exists():
        raise FileNotFoundError(f"Directory not found: {args.output_dir}")

    fit = load_stan_fit(args.output_dir)
    meta_path = args.output_dir / "windowed_detector_data.csv"
    if not meta_path.exists():
        windows_df = None
    else:
        windows_df = pd.read_csv(meta_path)
    
    print("Converting to InferenceData...")
    idata = az.from_cmdstanpy(
        posterior=fit,
        dims={
            "trend_season": ["time"],
            "trend_diel": ["time"],
            "lambda": ["time"]
        }
    )

    # --- New & Updated Diagnostics ---
    check_divergences(idata)
    plot_mcmc_health(idata, args.output_dir)
    plot_diagnostics(idata, args.output_dir)
    
    if windows_df is not None:
        plot_spline_components(fit, windows_df, args.output_dir)
        plot_total_intensity(fit, windows_df, args.output_dir)

    print(f"\n✅ Visualization complete. Check {args.output_dir} for PNGs.")

if __name__ == "__main__":
    main()