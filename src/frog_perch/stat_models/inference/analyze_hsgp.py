#!/usr/bin/env python3
"""
HSGP Call Intensity Posterior Analysis

- Loads merged detector CSVs
- Loads posterior draws from CmdStan CSVs
- Computes posterior predictive surfaces (seasonal + diel)
- Includes random effects u_day and u_minute
- Plots median + 90% CI
- Plots posterior hyperparameters
- Saves all plots to an output directory
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC, from_csv
from prepare_stan_data_hsgp import make_hsgp_basis


# ==============================
# ===== User Configurable ======
# ==============================
CSV_PATH = Path("stat_results/call_intensity_hsgp_v2/merged_detector_data.csv")
POSTERIOR_CSV_DIR = Path("stat_results/call_intensity_hsgp_v2")
POSTERIOR_PATTERN = "call_intensity_hsgp-*.csv"
OUTPUT_DIR = Path("stat_results/call_intensity_hsgp_v2/analysis_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

M_SEASON = 25
M_DIEL = 100
DAYS_MARGIN = 5
MINUTES_MARGIN = 60
QUANTILES = [0.05, 0.5, 0.95]  # 90% CI

# ==============================
# ===== Helper Functions =======
# ==============================

def load_fit(results_dir: Path, pattern: str = POSTERIOR_PATTERN) -> CmdStanMCMC:
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        raise RuntimeError(f"No CmdStan CSV files found in {results_dir} matching {pattern}")
    print(f"Loading {len(csvs)} chains from {results_dir}...")
    return from_csv([str(f) for f in csvs])

def compute_posterior_surfaces(fit: CmdStanMCMC, df_obs: pd.DataFrame):
    """Compute posterior predictive surfaces for seasonal/diel components and full surface."""
    df = df_obs.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    minutes_obs = (df["time_of_day_hours"]*60).astype(int)

    # Windowed grids
    day_grid = np.arange(df["day_of_year"].min()-DAYS_MARGIN,
                         df["day_of_year"].max()+DAYS_MARGIN+1)
    min_grid = np.arange(minutes_obs.min()-MINUTES_MARGIN,
                         minutes_obs.max()+MINUTES_MARGIN+1)

    X_s = make_hsgp_basis(day_grid, M=M_SEASON)
    X_m = make_hsgp_basis(min_grid, M=M_DIEL)

    # Extract draws
    draws = fit.draws_pd()
    n_draws = len(draws)

    # --- Seasonal ---
    beta_s_cols = [f"beta_season[{i+1}]" for i in range(2*M_SEASON)]
    mu_s = draws["mu_season"].to_numpy()[:, None]
    beta_s = draws[beta_s_cols].to_numpy()
    s_smooth = mu_s + beta_s @ X_s.T

    # Daily random effect
    u_day_std_cols = [f"u_day_std[{i+1}]" for i in range(df["day_of_year"].nunique())]
    u_day_std = draws[u_day_std_cols].to_numpy()
    sigma_day = draws["sigma_day_proc"].to_numpy()[:, None]
    u_day = sigma_day * u_day_std

    # Map u_day to grid days
    day_map = {val: i for i, val in enumerate(sorted(df["day_of_year"].unique()))}
    u_day_grid = np.zeros((n_draws, len(day_grid)))
    for i, d in enumerate(day_grid):
        if d in day_map:
            u_day_grid[:, i] = u_day[:, day_map[d]]

    s_with_day = s_smooth + u_day_grid
    p_smooth = 1/(1+np.exp(-s_smooth))
    p_with_day = 1/(1+np.exp(-s_with_day))

    # --- Diel ---
    beta_m_cols = [f"beta_diel[{i+1}]" for i in range(2*M_DIEL)]
    mu_m = draws["mu_diel"].to_numpy()[:, None]
    beta_m = draws[beta_m_cols].to_numpy()
    g_smooth = mu_m + beta_m @ X_m.T

    u_min_std_cols = [f"u_minute_std[{i+1}]" for i in range(len(min_grid))]
    # For minute effects, if exact indices not available, just skip u_min
    if all(col in draws.columns for col in u_min_std_cols):
        u_min_std = draws[u_min_std_cols].to_numpy()
        sigma_min = draws["sigma_minute"].to_numpy()[:, None]
        u_min_grid = u_min_std * sigma_min
        g_with_min = g_smooth + u_min_grid
    else:
        g_with_min = g_smooth

    p_diel_smooth = 1/(1+np.exp(-g_smooth))
    p_diel_with_min = 1/(1+np.exp(-g_with_min))

    # --- Full 2D surface ---
    surface = np.einsum('id,im->idm', p_smooth, p_diel_smooth)
    surface_mean = surface.mean(axis=0)

    return {
        "day_grid": day_grid,
        "min_grid": min_grid,
        "p_smooth": p_smooth,
        "p_with_day": p_with_day,
        "p_diel_smooth": p_diel_smooth,
        "p_diel_with_min": p_diel_with_min,
        "surface_mean": surface_mean,
        "draws": draws
    }

def plot_quantiles(x, draws, quantiles, xlabel, ylabel, title, fname):
    q_vals = {f"q{int(q*100)}": np.quantile(draws, q, axis=0) for q in quantiles}
    plt.figure(figsize=(10,5))
    plt.fill_between(x, q_vals["q5"], q_vals["q95"], alpha=0.5, color="lightgray", label="90% CI")
    plt.plot(x, q_vals["q50"], color="black", lw=1.5, label="Median")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=200)
    plt.close()

def plot_surface(day_grid, min_grid, surface, fname):
    plt.figure(figsize=(12,6))
    plt.imshow(surface.T, origin='lower',
               extent=[day_grid[0], day_grid[-1], min_grid[0], min_grid[-1]],
               aspect='auto', cmap='viridis')
    plt.colorbar(label="P(Call)")
    plt.xlabel("Day of Year")
    plt.ylabel("Minute of Day")
    plt.title("Posterior Predictive Call Probability Surface (Median)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=200)
    plt.close()

def plot_hyperparameters(draws):
    scalar_hyperparams = [
        "mu_season", "mu_diel",
        "alpha_season", "alpha_diel",
        "rho_season", "rho_diel",
        "sigma_day_proc", "sigma_minute"
    ]
    for param in scalar_hyperparams:
        if param not in draws.columns:
            continue
        plt.figure(figsize=(6,4))
        plt.hist(draws[param], bins=50, density=True, alpha=0.7, color="steelblue")
        plt.title(f"Posterior of {param}")
        plt.xlabel(param)
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR/f"{param}_posterior.png", dpi=200)
        plt.close()

# ==============================
# ===== Main Pipeline ==========
# ==============================

def main():
    print("Loading merged detector CSV...")
    df = pd.read_csv(CSV_PATH)

    print("Loading posterior draws...")
    fit = load_fit(POSTERIOR_CSV_DIR, POSTERIOR_PATTERN)

    print("Computing posterior surfaces...")
    post = compute_posterior_surfaces(fit, df)

    print("Plotting seasonal curves...")
    plot_quantiles(post["day_grid"], post["p_smooth"], QUANTILES, "Day of Year", "P(Call)", "Seasonal Probability (Smooth)", "seasonal_smooth.png")
    plot_quantiles(post["day_grid"], post["p_with_day"], QUANTILES, "Day of Year", "P(Call)", "Seasonal Probability + Daily Random Effects", "seasonal_with_day.png")

    print("Plotting diel curves...")
    plot_quantiles(post["min_grid"]/60, post["p_diel_smooth"], QUANTILES, "Hour of Day", "P(Call)", "Diel Probability (Smooth)", "diel_smooth.png")
    plot_quantiles(post["min_grid"]/60, post["p_diel_with_min"], QUANTILES, "Hour of Day", "P(Call)", "Diel Probability + Minute Random Effects", "diel_with_min.png")

    print("Plotting 2D posterior surface...")
    plot_surface(post["day_grid"], post["min_grid"], post["surface_mean"], "call_surface_heatmap.png")

    print("Plotting scalar hyperparameters...")
    plot_hyperparameters(post["draws"])

    print(f"All plots saved to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
