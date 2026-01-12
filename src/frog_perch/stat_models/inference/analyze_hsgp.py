#!/usr/bin/env python3
"""
Analysis script for call_intensity_hsgp (HSGP Fourier model).

- Reconstructs FULL seasonal and diel curves using HSGP spectral densities.
- Produces unified diagnostics for HSGP hyperparameters (alpha, rho) and error scales.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC, from_csv

# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

RESULTS_DIR = Path("stat_results/call_intensity_hsgp_v1")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"
CSV_GLOB_PATTERN = "call_intensity_hsgp-*.csv"

# HSGP Config (MUST MATCH STAN DATA!)
M_SEASON = 50
M_DIEL   = 50

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
        # Normalize observed_x if plotting hours (0-24) but data is minutes (0-1440)
        # We assume the caller handles unit matching, but we'll check range
        rug_x = observed_x
        if xlabel == "Hour of Day" and observed_x.max() > 24:
             rug_x = observed_x / 60.0
             
        plt.scatter(rug_x, np.full_like(rug_x, summary["q025"].min()), 
                    marker='|', color='red', alpha=0.3, s=10, label="Data Present")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# HSGP Reconstruction Logic
# ---------------------------------------------------------------------

def make_fourier_design_matrix(x_vals: np.ndarray, period: float, M: int) -> np.ndarray:
    """
    Generates a Fourier design matrix.
    """
    N = len(x_vals)
    X = np.zeros((N, 2 * M))
    for m in range(1, M + 1):
        omega = 2.0 * np.pi * m / period
        X[:, 2 * (m - 1)]     = np.cos(omega * x_vals)
        X[:, 2 * (m - 1) + 1] = np.sin(omega * x_vals)
    return X


def compute_diagSPD(alpha: np.ndarray, rho: np.ndarray, period: float, M: int) -> np.ndarray:
    """
    Computes the Spectral Density vector for Squared Exponential (RBF).
    S(w) = alpha^2 * sqrt(2*pi) * rho * exp(-0.5 * rho^2 * w^2)
    """
    n_draws = len(alpha)
    diagSPD = np.zeros((n_draws, 2 * M))
    
    # Precompute constant term: alpha^2 * sqrt(2pi) * rho
    # alpha, rho are shape (n_draws,)
    term1 = (alpha**2) * np.sqrt(2.0 * np.pi) * rho

    for m in range(1, M + 1):
        w = 2.0 * np.pi * m / period
        
        # S(w) formula for RBF
        S = term1 * np.exp(-0.5 * (rho * w)**2)
        
        sqrt_S = np.sqrt(S)
        
        # Fill both Cos and Sin slots
        diagSPD[:, 2*(m-1)]     = sqrt_S
        diagSPD[:, 2*(m-1) + 1] = sqrt_S
        
    return diagSPD


def reconstruct_curves(fit: CmdStanMCMC, X_season_full, X_diel_full, M_season, M_diel):
    """
    Computes curve = mu + X * (beta * diagSPD) for every draw.
    """
    # 1. Extract Parameters
    mu_season    = fit.stan_variable("mu_season")     # (draws,)
    beta_season  = fit.stan_variable("beta_season")   # (draws, 2*M)
    alpha_season = fit.stan_variable("alpha_season")  # (draws,)
    rho_season   = fit.stan_variable("rho_season")    # (draws,)
    
    mu_diel      = fit.stan_variable("mu_diel")
    beta_diel    = fit.stan_variable("beta_diel")
    alpha_diel   = fit.stan_variable("alpha_diel")
    rho_diel     = fit.stan_variable("rho_diel")

    # 2. Compute Spectral Densities (Scaling Factors)
    # shape: (draws, 2*M)
    spd_season = compute_diagSPD(alpha_season, rho_season, 365.0, M_season)
    spd_diel   = compute_diagSPD(alpha_diel,   rho_diel,   1440.0, M_diel)

    # 3. Apply Scaling (Non-Centered Parameterization)
    # beta_scaled = beta_raw * sqrt(S)
    beta_season_scaled = beta_season * spd_season
    beta_diel_scaled   = beta_diel   * spd_diel

    # 4. Project to Time Domain
    # s = mu + X @ beta_scaled.T
    # (draws, 1) + (draws, 2M) @ (2M, T) -> (draws, T)
    
    # Einsum: 'dk, tk -> dt' 
    # d=draws, k=basis_funcs, t=time_points
    s_logit = mu_season[:, None] + np.einsum('dk,tk->dt', beta_season_scaled, X_season_full)
    p_season = inv_logit(s_logit)

    g_logit = mu_diel[:, None]   + np.einsum('dk,tk->dt', beta_diel_scaled,   X_diel_full)
    p_diel = inv_logit(g_logit)

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

    # Posterior of Error Scales & Hyperparameters
    params_to_plot = [
        "sigma_day_proc", "sigma_ell", 
        "alpha_season", "rho_season", 
        "alpha_diel", "rho_diel"
    ]
    
    for param in params_to_plot:
        try:
            data = fit.stan_variable(param)
            plt.figure(figsize=(6,4))
            plt.hist(data, bins=50, density=True, alpha=0.7)
            plt.title(f"Posterior: {param}")
            plt.grid(alpha=0.3)
            plt.tight_layout()
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
            
    # 3. Generate Full Design Matrices
    print("Generating full Fourier design matrices...")
    full_days = np.arange(1, 366)
    full_mins = np.arange(0, 1440)
    
    X_s = make_fourier_design_matrix(full_days, 365.0, M_SEASON)
    X_d = make_fourier_design_matrix(full_mins, 1440.0, M_DIEL)
    
    # 4. Reconstruct Curves
    print("Reconstructing HSGP curves from posterior...")
    p_season_draws, p_diel_draws = reconstruct_curves(fit, X_s, X_d, M_SEASON, M_DIEL)
    
    summ_season = summarize_curve(p_season_draws)
    summ_diel   = summarize_curve(p_diel_draws)

    # 5. Plotting
    print("Plotting...")
    plot_diagnostics(fit, RESULTS_DIR)

    plot_with_bands(
        full_days, summ_season, 
        "Seasonal Probability (HSGP)", 
        "Day of Year", "P(Call)", 
        RESULTS_DIR / "seasonal_curve.png",
        observed_x=obs_days
    )

    plot_with_bands(
        full_mins / 60.0, summ_diel, 
        "Diel Probability (HSGP)", 
        "Hour of Day", "P(Call)", 
        RESULTS_DIR / "diel_curve.png",
        observed_x=obs_mins
    )

    print(f"Done! Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()