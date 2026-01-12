#!/usr/bin/env python3
"""
Analysis script for call_intensity_spline (Cyclic P-Spline model).

- Reconstructs FULL seasonal and diel curves from spline weights (beta).
- Produces unified diagnostics for the new parameters (betas, sigma_ell).
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC, from_csv
from patsy import dmatrix  # REQUIRED for spline reconstruction

# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

# Update these paths to match your new output directory
RESULTS_DIR = Path("stat_results/call_intensity_spline_windowed_v1")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"
CSV_GLOB_PATTERN = "call_intensity_spline_windowed-*.csv"

# Spline Config (MUST MATCH STAN DATA!)
K_SEASON = 50
K_DIEL   = 50

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def load_fit(results_dir: Path, pattern: str) -> CmdStanMCMC:
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        raise RuntimeError(f"No CmdStan CSV files found in {results_dir} matching {pattern}")
    print(f"Loading {len(csvs)} chains from {results_dir}...")
    return from_csv([str(f) for f in csvs])

def summarize_curve(draws_curve: np.ndarray) -> dict:
    q = np.quantile(draws_curve, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    return {
        "mean":   draws_curve.mean(axis=0),
        "q025":   q[0],
        "q25":    q[1],
        "median": q[2],
        "q75":    q[3],
        "q975":   q[4],
    }

def plot_with_bands(x: np.ndarray, summary: dict, title: str, xlabel: str, ylabel: str, out_path: Path, observed_x: np.ndarray = None):
    plt.figure(figsize=(10, 5))
    
    # Plot bands
    plt.fill_between(x, summary["q025"], summary["q975"], color="lightgray", alpha=0.5, label="95% CI")
    plt.fill_between(x, summary["q25"], summary["q75"], color="gray", alpha=0.5, label="50% CI")
    plt.plot(x, summary["median"], color="black", lw=1.5, label="Median")

    # Add Rug Plot for data presence if provided
    if observed_x is not None:
        plt.scatter(observed_x, np.full_like(observed_x, summary["q025"].min()), 
                    marker='|', color='red', alpha=0.3, s=10, label="Data Present")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# Spline Reconstruction Logic
# ---------------------------------------------------------------------

def get_full_design_matrices(k_season: int, k_diel: int):
    """
    Generates the FULL design matrices for plotting (all 365 days, all 1440 mins).
    """
    # Match the window defined in Data Prep
    days = np.arange(240, 321)  # Aug 28 - Nov 16
    mins = np.arange(17*60, 21*60 + 1) # 17:00 - 21:00
    
    # Use 'bs' (B-spline), not 'cc'
    X_s = dmatrix(f"bs(x, df={k_season}, degree=3, include_intercept=True) - 1", 
                  {"x": days}, return_type='dataframe').values
                  
    X_d = dmatrix(f"bs(x, df={k_diel}, degree=3, include_intercept=True) - 1", 
                  {"x": mins}, return_type='dataframe').values
                  
    return X_s, X_d, days, mins


def reconstruct_curves(fit: CmdStanMCMC, X_season_full, X_diel_full):
    """
    Computes curve = mu + X * beta for every draw.
    """
    # Extract params
    mu_season   = fit.stan_variable("mu_season")      # (draws,)
    beta_season = fit.stan_variable("beta_season")    # (draws, K_season)
    
    mu_diel     = fit.stan_variable("mu_diel")        # (draws,)
    beta_diel   = fit.stan_variable("beta_diel")      # (draws, K_diel)

    # --- Seasonal Reconstruction ---
    # shape: (draws, 1) + (draws, K) @ (K, 365) -> (draws, 365)
    # Note: einsum is safer for batch matmul
    # 'dk, tk -> dt' means (draws, knots) * (time, knots).T
    
    s_logit = mu_season[:, None] + np.einsum('dk,tk->dt', beta_season, X_season_full)
    p_season = inv_logit(s_logit)

    # --- Diel Reconstruction ---
    g_logit = mu_diel[:, None]   + np.einsum('dk,tk->dt', beta_diel,   X_diel_full)
    p_diel   = inv_logit(g_logit)

    return p_season, p_diel

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def plot_diagnostics(fit: CmdStanMCMC, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    summ = fit.summary()
    
    # Histograms of Rhat/ESS
    for metric in ["R_hat", "ESS_bulk", "ESS_tail"]:
        plt.figure(figsize=(6, 4))
        plt.hist(summ[metric], bins=30)
        plt.title(f"Histogram of {metric}")
        plt.savefig(outdir / f"hist_{metric}.png")
        plt.close()

    # Posterior of Error Scales
    for param in ["sigma_day_proc", "sigma_ell", "tau_season", "tau_diel"]:
        try:
            data = fit.stan_variable(param)
            plt.figure()
            plt.hist(data, bins=50, density=True, alpha=0.7)
            plt.title(f"Posterior: {param}")
            plt.savefig(outdir / f"post_{param}.png")
            plt.close()
        except Exception:
            print(f"Skipping {param} (not found)")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory {RESULTS_DIR} not found.")
        return

    # 1. Load Model
    fit = load_fit(RESULTS_DIR, CSV_GLOB_PATTERN)
    
    # 2. Load Observed Data (for Rug Plot)
    obs_days = None
    obs_mins = None
    if MERGED_CSV.exists():
        df = pd.read_csv(MERGED_CSV)
        if "day_of_year" in df.columns:
            obs_days = df["day_of_year"].unique()
        if "minute_of_day" in df.columns:
            obs_mins = df["minute_of_day"].unique()
            
    # 3. Generate Design Matrices
    print("Generating full design matrices...")
    X_s, X_d, days, mins = get_full_design_matrices(K_SEASON, K_DIEL)
    
    # 4. Reconstruct Curves
    print("Reconstructing curves from posterior...")
    p_season_draws, p_diel_draws = reconstruct_curves(fit, X_s, X_d)
    
    summ_season = summarize_curve(p_season_draws)
    summ_diel   = summarize_curve(p_diel_draws)

    # 5. Plotting
    print("Plotting...")
    plot_diagnostics(fit, RESULTS_DIR)

    plot_with_bands(
        days, summ_season, 
        "Seasonal Probability (Smooth Spline)", 
        "Day of Year", "P(Call)", 
        RESULTS_DIR / "seasonal_curve.png",
        observed_x=obs_days
    )

    plot_with_bands(
        mins / 60.0, summ_diel, 
        "Diel Probability (Smooth Spline)", 
        "Hour of Day", "P(Call)", 
        RESULTS_DIR / "diel_curve.png",
        observed_x=(obs_mins / 60.0) if obs_mins is not None else None
    )

    print(f"Done! Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()