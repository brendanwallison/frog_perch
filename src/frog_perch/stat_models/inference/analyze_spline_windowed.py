#!/usr/bin/env python3
"""
Analysis script for call_intensity_window (Non-Cyclic B-Spline model).

- Reconstructs seasonal and diel curves across the specific observed window.
- Uses FIXED LINEAR KNOTS to match the Stan data prep exactly.
- Handles day-level and minute-level nugget variance.
- Saves plots for both dense grid and observed points.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC, from_csv
from patsy import dmatrix
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------

RESULTS_DIR = Path("stat_results/call_intensity_spline_windowed_v1")
MERGED_CSV = RESULTS_DIR / "merged_detector_data.csv"
CSV_GLOB_PATTERN = "call_intensity_spline_windowed-*.csv"

# --- WINDOW DEFINITIONS (Must match Data Prep) ---
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

def plot_with_bands(x: np.ndarray, summary: dict, title: str, xlabel: str, ylabel: str, out_path: Path, observed_x: np.ndarray = None):
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, summary["q025"], summary["q975"], color="lightgray", alpha=0.5, label="95% CI")
    plt.fill_between(x, summary["q25"], summary["q75"], color="gray", alpha=0.5, label="50% CI")
    plt.plot(x, summary["median"], color="black", lw=1.5, label="Median")

    if observed_x is not None:
        rug_x = observed_x
        if xlabel == "Hour of Day" and observed_x.max() > 24:
            rug_x = observed_x / 60.0
        mask = (rug_x >= x.min()) & (rug_x <= x.max())
        plt.scatter(rug_x[mask], np.full_like(rug_x[mask], summary["q025"].min()), 
                    marker='|', color='red', alpha=0.3, s=10, label="Data Present")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# ---------------------------------------------------------------------
# Robust B-Spline Matrix Logic
# ---------------------------------------------------------------------

def get_linear_bspline_matrix(x_vals, min_val, max_val, n_dofs, degree=3):
    n_inner = n_dofs - degree - 1
    if n_inner < 0:
        return dmatrix(
            "bs(x, df=n_dofs, degree=degree, include_intercept=True, lower_bound=min_val, upper_bound=max_val) - 1",
            {"x": x_vals, "n_dofs": n_dofs, "degree": degree, "min_val": min_val, "max_val": max_val},
            return_type='dataframe'
        ).values
    knots = np.linspace(min_val, max_val, n_inner + 2)[1:-1]
    return dmatrix(
        "bs(x, knots=knots, degree=degree, include_intercept=True, lower_bound=min_val, upper_bound=max_val) - 1",
        {"x": x_vals, "knots": knots, "degree": degree, "min_val": min_val, "max_val": max_val},
        return_type='dataframe'
    ).values

# ---------------------------------------------------------------------
# Curve Reconstruction
# ---------------------------------------------------------------------

def reconstruct_curves(fit, X_season_grid, X_diel_grid, obs_days, obs_mins, merged_df):
    """
    Reconstruct posterior curves for season and diel probabilities.
    """

    # Extract parameters
    mu_season   = fit.stan_variable("mu_season")
    beta_season = fit.stan_variable("beta_season")
    mu_diel     = fit.stan_variable("mu_diel")
    beta_diel   = fit.stan_variable("beta_diel")
    u_day       = fit.stan_variable("u_day")
    u_minute    = fit.stan_variable("u_minute")

    n_draws = beta_season.shape[0]

    # Center nuggets
    u_day_centered = u_day - u_day.mean(axis=1, keepdims=True)
    u_min_centered = u_minute - u_minute.mean(axis=1, keepdims=True)

    # --- Dense grid linear predictor ---
    s_grid = mu_season[:, None] + beta_season @ X_season_grid.T
    g_grid = mu_diel[:, None] + beta_diel @ X_diel_grid.T

    n_days_grid = X_season_grid.shape[0]
    n_mins_grid = X_diel_grid.shape[0]

    p_season_grid = np.empty_like(s_grid)
    p_diel_grid   = np.empty_like(g_grid)

    for i in range(n_draws):
        f_day = interp1d(np.arange(u_day_centered.shape[1]), u_day_centered[i], kind="linear", fill_value="extrapolate")
        f_min = interp1d(np.arange(u_min_centered.shape[1]), u_min_centered[i], kind="linear", fill_value="extrapolate")
        u_day_grid = f_day(np.linspace(0, u_day_centered.shape[1]-1, n_days_grid))
        u_min_grid = f_min(np.linspace(0, u_min_centered.shape[1]-1, n_mins_grid))
        p_season_grid[i] = inv_logit(s_grid[i] + u_day_grid)
        p_diel_grid[i]   = inv_logit(g_grid[i] + u_min_grid)

    # --- Observed points ---
    X_season_obs = get_linear_bspline_matrix(obs_days, SEASON_START, SEASON_END, beta_season.shape[1])
    X_diel_obs   = get_linear_bspline_matrix(obs_mins, MIN_START, MIN_END, beta_diel.shape[1])
    s_obs = mu_season[:, None] + beta_season @ X_season_obs.T
    g_obs = mu_diel[:, None] + beta_diel @ X_diel_obs.T

    n_days_obs = len(obs_days)
    n_mins_obs = len(obs_mins)
    p_season_obs = np.empty_like(s_obs)
    p_diel_obs   = np.empty_like(g_obs)

    for i in range(n_draws):
        f_day = interp1d(np.arange(u_day_centered.shape[1]), u_day_centered[i], kind="linear", fill_value="extrapolate")
        f_min = interp1d(np.arange(u_min_centered.shape[1]), u_min_centered[i], kind="linear", fill_value="extrapolate")
        u_day_obs = f_day(np.linspace(0, u_day_centered.shape[1]-1, n_days_obs))
        u_min_obs = f_min(np.linspace(0, u_min_centered.shape[1]-1, n_mins_obs))
        p_season_obs[i] = inv_logit(s_obs[i] + u_day_obs)
        p_diel_obs[i]   = inv_logit(g_obs[i] + u_min_obs)

    return p_season_obs, p_diel_obs, p_season_grid, p_diel_grid

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
    beta_season = fit.stan_variable("beta_season")
    beta_diel   = fit.stan_variable("beta_diel")
    print(f"Detected Dimensions: K_season={beta_season.shape[1]}, K_diel={beta_diel.shape[1]}")

    # Load merged CSV for observed points
    df = pd.read_csv(MERGED_CSV)
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear.astype(int)
    df["minute_of_day"] = (df["time_of_day_hours"] * 60).astype(int)

    # Clip to spline boundaries
    obs_days = np.clip(np.sort(df["day_of_year"].unique()), SEASON_START, SEASON_END)
    obs_mins = np.clip(np.sort(df["minute_of_day"].unique()), MIN_START, MIN_END)

    # Dense plotting grids
    plot_days = np.linspace(SEASON_START, SEASON_END, 200)
    plot_mins = np.linspace(MIN_START, MIN_END, 200)
    X_s_grid = get_linear_bspline_matrix(plot_days, SEASON_START, SEASON_END, beta_season.shape[1])
    X_d_grid = get_linear_bspline_matrix(plot_mins, MIN_START, MIN_END, beta_diel.shape[1])

    # Reconstruct posterior curves
    p_season_obs, p_diel_obs, p_season_grid, p_diel_grid = reconstruct_curves(
        fit, X_s_grid, X_d_grid, obs_days, obs_mins, df
    )

    # Summarize
    summ_season_obs = summarize_curve(p_season_obs)
    summ_diel_obs   = summarize_curve(p_diel_obs)
    summ_season_grid = summarize_curve(p_season_grid)
    summ_diel_grid   = summarize_curve(p_diel_grid)

    # Plot diagnostics
    plot_diagnostics(fit, RESULTS_DIR, include_minute=True)

    # Plot curves
    plot_with_bands(plot_days, summ_season_grid, "Seasonal Probability (Grid)", "Day of Year", "P(Call)",
                    RESULTS_DIR / "seasonal_curve_grid.png", observed_x=obs_days)
    plot_with_bands(obs_days, summ_season_obs, "Seasonal Probability (Observed)", "Day of Year", "P(Call)",
                    RESULTS_DIR / "seasonal_curve_obs.png", observed_x=obs_days)
    plot_with_bands(plot_mins / 60.0, summ_diel_grid, "Diel Probability (Grid)", "Hour of Day", "P(Call)",
                    RESULTS_DIR / "diel_curve_grid.png", observed_x=obs_mins)
    plot_with_bands(obs_mins / 60.0, summ_diel_obs, "Diel Probability (Observed)", "Hour of Day", "P(Call)",
                    RESULTS_DIR / "diel_curve_obs.png", observed_x=obs_mins)

    print(f"Done! Results in {RESULTS_DIR}")

if __name__ == "__main__":
    main()
