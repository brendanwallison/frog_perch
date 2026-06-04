#!/usr/bin/env python3

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from cmdstanpy import CmdStanMCMC, from_csv

# ==============================
# ===== HSGP Logic (Base) ======
# ==============================
try:
    from prepare_stan_data_hsgp import make_hsgp_basis
except ImportError:
    # Use the inline version from your snippet
    def make_hsgp_basis(x: np.ndarray, M: int, L: float = None, c: float = 1.2, **kwargs) -> np.ndarray:
        x_min, x_max = x.min(), x.max()
        x_center = (x_min + x_max) / 2.0
        x_scale = (x_max - x_min) / 2.0
        if L is None: L = c 
        x_scaled = (x - x_center) / x_scale
        m_seq = np.arange(1, M + 1)
        arg = (np.pi * m_seq * (x_scaled[:, None] + L)) / (2 * L)
        phi = (1.0 / np.sqrt(L)) * np.sin(arg)
        return phi

# ==============================
# ===== User Configurable ======
# ==============================
CSV_PATH = Path("stat_results/call_intensity_hsgp_v2/merged_detector_data.csv")
POSTERIOR_CSV_DIR = Path("stat_results/call_intensity_hsgp_v2")
POSTERIOR_PATTERN = "call_intensity_hsgp-*.csv"
OUTPUT_DIR = Path("stat_results/call_intensity_hsgp_v2/analysis_output_binned")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

M_SEASON = 25
M_DIEL = 100
DAYS_MARGIN = 5
QUANTILES = [0.05, 0.5, 0.95]

# Rate Conversion Config
SLICE_DURATION_SEC = 5.0 / 16.0  # 0.3125 seconds
SLICES_PER_MINUTE = 60.0 / SLICE_DURATION_SEC  # 192.0

# Binning Config (for narrower CIs)
BIN_WIDTH_MINUTES = 30  # Average rate over 30 min blocks
BIN_WIDTH_DAYS = 7      # Average seasonal rate over 1 week

# ==============================
# ===== Helper Functions =======
# ==============================

def load_fit(results_dir: Path, pattern: str) -> CmdStanMCMC:
    csvs = sorted(results_dir.glob(pattern))
    if not csvs:
        raise RuntimeError(f"No CmdStan CSV files found in {results_dir} matching {pattern}")
    print(f"Loading {len(csvs)} chains...")
    return from_csv([str(f) for f in csvs])

def compute_posterior_rates(fit: CmdStanMCMC, df_obs: pd.DataFrame):
    df = df_obs.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["day_of_year"] = df["date"].dt.dayofyear
    
    # --- 1. Define High-Res Grids ---
    # We use explicit physical domains (1-365, 0-1440) to ensuring the internal 
    # scaling in make_hsgp_basis works correctly.
    
    # Season: Day 1 to 365
    day_grid = np.arange(df["day_of_year"].min() - DAYS_MARGIN,
                         df["day_of_year"].max() + DAYS_MARGIN + 1)
    
    # Diel: Minute 0 to 1440 (24 hours) - 1 minute resolution
    min_grid = np.arange(0, 1441, 1) 

    # --- 2. Build HSGP Basis ---
    X_s = make_hsgp_basis(day_grid, M=M_SEASON)
    X_m = make_hsgp_basis(min_grid, M=M_DIEL)

    draws = fit.draws_pd()
    
    # --- 3. Compute Probabilities (Latent Trend) ---
    # Season
    mu_s = draws["mu_season"].to_numpy()[:, None]
    beta_s_cols = [f"beta_season[{i+1}]" for i in range(2 * M_SEASON)]
    beta_s = draws[beta_s_cols].to_numpy()
    p_season = 1/(1+np.exp(-(mu_s + beta_s @ X_s.T))) # Sigmoid

    # Diel
    mu_m = draws["mu_diel"].to_numpy()[:, None]
    beta_m_cols = [f"beta_diel[{i+1}]" for i in range(2 * M_DIEL)]
    beta_m = draws[beta_m_cols].to_numpy()
    p_diel = 1/(1+np.exp(-(mu_m + beta_m @ X_m.T)))   # Sigmoid

    # --- 4. Full Rate Surface ---
    # Prob(Season) * Prob(Diel) * SlicesPerMin
    prob_surface = np.einsum('id,im->idm', p_season, p_diel)
    rate_surface = prob_surface * SLICES_PER_MINUTE
    rate_med = np.median(rate_surface, axis=0)

    # --- 5. Marginals (converted to Rate) ---
    rate_season = p_season * SLICES_PER_MINUTE
    rate_diel = p_diel * SLICES_PER_MINUTE

    return {
        "day_grid": day_grid,
        "min_grid": min_grid,
        "rate_season": rate_season,
        "rate_diel": rate_diel,
        "rate_surface_med": rate_med,
        "draws": draws
    }

def bin_draws(grid, draws, bin_width):
    """Resamples draws to coarser resolution by averaging."""
    n_draws, n_time = draws.shape
    n_bins = n_time // bin_width
    trim_len = n_bins * bin_width
    
    draws_trimmed = draws[:, :trim_len]
    grid_trimmed = grid[:trim_len]
    
    # Reshape (N, Bins, Width) -> Mean(axis=2)
    draws_binned = draws_trimmed.reshape(n_draws, n_bins, bin_width).mean(axis=2)
    grid_binned = grid_trimmed.reshape(n_bins, bin_width).mean(axis=1)
    
    return grid_binned, draws_binned

def plot_binned_comparison(grid, draws, bin_width, xlabel, title, fname):
    """Plots High-Res vs Binned-Res CIs."""
    # 1. High Res
    q_high = np.quantile(draws, QUANTILES, axis=0)
    
    # 2. Binned
    grid_bin, draws_bin = bin_draws(grid, draws, bin_width)
    q_bin = np.quantile(draws_bin, QUANTILES, axis=0)
    
    plt.figure(figsize=(10, 5))
    
    # Plot Instantaneous (Gray)
    plt.fill_between(grid, q_high[0], q_high[2], color="gray", alpha=0.2, label="Instantaneous CI")
    
    # Plot Aggregated (Blue)
    plt.fill_between(grid_bin, q_bin[0], q_bin[2], color="tab:blue", alpha=0.5, label=f"{bin_width}-unit Avg CI")
    plt.plot(grid_bin, q_bin[1], color="black", linewidth=2, label="Median")
    
    plt.xlabel(xlabel)
    plt.ylabel("Calls / Minute")
    plt.title(f"{title}\n(Gray = Instantaneous, Blue = {bin_width}-unit Average)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=200)
    plt.close()

def plot_surface(day_grid, min_grid, surface_med, fname):
    plt.figure(figsize=(12,5))
    plt.imshow(surface_med.T, origin="lower", aspect="auto",
               extent=[day_grid[0], day_grid[-1], min_grid[0], min_grid[-1]],
               cmap="inferno")
    plt.colorbar(label="Expected Calls / Minute")
    plt.xlabel("Day of Year"); plt.ylabel("Minute of Day")
    plt.title("Expected Call Rate Surface")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/fname, dpi=200); plt.close()

def plot_arviz_diagnostics(fit):
    print("Generating ArviZ diagnostics...")
    idata = az.from_cmdstanpy(posterior=fit)
    vars_to_plot = ["mu_season", "mu_diel", "alpha_season", "alpha_diel", "rho_season", "rho_diel", "sigma_det"]
    vars_to_plot = [v for v in vars_to_plot if v in idata.posterior.data_vars]

    if vars_to_plot:
        az.summary(idata, var_names=vars_to_plot).to_csv(OUTPUT_DIR / "arviz_summary.csv")
        az.plot_trace(idata, var_names=vars_to_plot)
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "arviz_trace.png", dpi=150); plt.close()

# ==============================
# ========= Main ===============
# ==============================

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    fit = load_fit(POSTERIOR_CSV_DIR, POSTERIOR_PATTERN)
    
    plot_arviz_diagnostics(fit)

    print("Computing posterior rates...")
    post = compute_posterior_rates(fit, df)

    # 1. Diel Plot (Binning Minutes)
    print(f"Plotting Diel Rate (Binning: {BIN_WIDTH_MINUTES} mins)...")
    plot_binned_comparison(
        post["min_grid"]/60, 
        post["rate_diel"], 
        BIN_WIDTH_MINUTES,
        xlabel="Hour of Day", 
        title="Diel Call Rate: Instantaneous vs Averaged",
        fname="diel_rate_binned.png"
    )

    # 2. Season Plot (Binning Days)
    print(f"Plotting Seasonal Rate Potential (Binning: {BIN_WIDTH_DAYS} days)...")
    plot_binned_comparison(
        post["day_grid"], 
        post["rate_season"], 
        BIN_WIDTH_DAYS,
        xlabel="Day of Year", 
        title="Seasonal Potential: Daily vs Weekly Avg",
        fname="seasonal_rate_binned.png"
    )

    # 3. Surface Plot
    print("Plotting 2D Surface...")
    plot_surface(post["day_grid"], post["min_grid"], post["rate_surface_med"],
                 "call_rate_surface.png")

    print(f"Done. Saved to {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()