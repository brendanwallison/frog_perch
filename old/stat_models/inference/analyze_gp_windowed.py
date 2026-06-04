#!/usr/bin/env python3
"""
Analysis script for call_intensity_window (GP model replacement).

- Reconstructs seasonal and diel curves at the GP latent points (no interpolation).
- Only grid predictions are generated; no observed-point reconstruction.
- Saves plots for grid points only.
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

RESULTS_DIR = Path("stat_results/call_intensity_spline_windowed_v1")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"
CSV_GLOB_PATTERN = "call_intensity_spline_windowed-*.csv"

SEASON_START = 240   # Aug 28
SEASON_END   = 320   # Nov 16
MIN_START    = 17 * 60
MIN_END      = 21 * 60

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

def plot_with_bands(x: np.ndarray, summary: dict, title: str, xlabel: str, ylabel: str, out_path: Path):
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, summary["q025"], summary["q975"], color="lightgray", alpha=0.5, label="95% CI")
    plt.fill_between(x, summary["q25"], summary["q75"], color="gray", alpha=0.5, label="50% CI")
    plt.plot(x, summary["median"], color="black", lw=1.5, label="Median")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# Grid Reconstruction (GP latent points)
# ---------------------------------------------------------------------

def reconstruct_grid(fit):
    """
    Reconstruct posterior curves at GP latent points only.
    """

    mu_season   = fit.stan_variable("mu_season")
    mu_diel     = fit.stan_variable("mu_diel")
    u_day       = fit.stan_variable("u_day")      # (n_draws, D_latent)
    u_minute    = fit.stan_variable("u_minute")   # (n_draws, M_latent)

    n_draws = u_day.shape[0]
    D_latent = u_day.shape[1]
    M_latent = u_minute.shape[1]

    # Center nuggets
    u_day_centered = u_day - u_day.mean(axis=1, keepdims=True)
    u_min_centered = u_minute - u_minute.mean(axis=1, keepdims=True)

    # Posterior probability curves at latent points
    p_season_grid = inv_logit(mu_season[:, None] + u_day_centered)
    p_diel_grid   = inv_logit(mu_diel[:, None] + u_min_centered)

    return p_season_grid, p_diel_grid

# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def plot_diagnostics(fit: CmdStanMCMC, outdir: Path, include_minute=False):
    outdir.mkdir(parents=True, exist_ok=True)
    summ = fit.summary()
    for metric in ["R_hat", "ESS_bulk", "ESS_tail"]:
        plt.figure(figsize=(6,4))
        plt.hist(summ[metric], bins=30)
        plt.title(f"Histogram of {metric}")
        plt.savefig(outdir / f"hist_{metric}.png")
        plt.close()

    params_to_plot = ["sigma_day_proc", "tau_season", "tau_diel"]
    if include_minute:
        params_to_plot.append("sigma_minute")

    for param in params_to_plot:
        try:
            data = fit.stan_variable(param)
            plt.figure(figsize=(6,4))
            plt.hist(data, bins=50, density=True, alpha=0.7)
            plt.title(f"Posterior: {param}")
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

    # Load model
    fit = load_fit(RESULTS_DIR, CSV_GLOB_PATTERN)
    u_day = fit.stan_variable("u_day")
    u_minute = fit.stan_variable("u_minute")
    print(f"Detected latent points: D_latent={u_day.shape[1]}, M_latent={u_minute.shape[1]}")

    # Grid values = latent points
    df = pd.read_csv(MERGED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)

    # Reconstruct posterior curves at latent points
    p_season_grid, p_diel_grid = reconstruct_grid(fit)

    # Summarize
    summ_season_grid = summarize_curve(p_season_grid)
    summ_diel_grid   = summarize_curve(p_diel_grid)

    # Plot diagnostics
    plot_diagnostics(fit, RESULTS_DIR, include_minute=True)

    # Plot curves
    plot_days = np.linspace(SEASON_START, SEASON_END, p_season_grid.shape[1])
    plot_mins = np.linspace(MIN_START, MIN_END, p_diel_grid.shape[1])

    plot_with_bands(plot_days, summ_season_grid, "Seasonal Probability (Grid)", "Day of Year", "P(Call)",
                    RESULTS_DIR / "seasonal_curve_grid.png")
    plot_with_bands(plot_mins / 60.0, summ_diel_grid, "Diel Probability (Grid)", "Hour of Day", "P(Call)",
                    RESULTS_DIR / "diel_curve_grid.png")

    print(f"Done! Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
