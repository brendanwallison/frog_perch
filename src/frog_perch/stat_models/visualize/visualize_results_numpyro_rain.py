#!/usr/bin/env python3
"""
Memory-safe, High-Performance Visualization for Hydrological Decay Model.
Universal source of truth for model reconstruction and temporal alignment.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

plt.style.use("ggplot")

# ============================================================
# Centralized Data Preprocessing & Alignment
# ============================================================

def preprocess_viz_metadata(windows_df, precip_daily, burn_in_days):
    window_dates = pd.to_datetime(windows_df["start_time"]).dt.normalize()
    audio_start = window_dates.min()
    sim_start = audio_start - pd.Timedelta(days=burn_in_days)
    
    print(f"--- Debugging Timeline ---")
    print(f"First Audio Window:   {audio_start.date()}")
    print(f"Calculated Sim Start: {sim_start.date()}")
    print(f"Precip Data Points:   {len(precip_daily)}")
    print(f"---------------------------")

    day_idx = (window_dates - sim_start).dt.days.values
    unique_window_dates = sorted(window_dates.unique().date)
    full_calendar = pd.date_range(start=sim_start, periods=len(precip_daily), freq="D")

    return {
        "day_idx": day_idx,
        "unique_window_dates": unique_window_dates,
        "full_calendar": full_calendar,
        "sim_start": sim_start,
        "burn_in_days": burn_in_days
    }

# ============================================================
# Model Reconstruction Utilities
# ============================================================

def get_param(container, name):
    """Handle transformed and raw param naming in Dataset or Draw dict."""
    # Check if we are dealing with an ArviZ posterior Dataset
    if hasattr(container, "data_vars"):
        val = container.get(f"{name}_val", container.get(name))
    else:
        # Dealing with a single draw dictionary
        val = container.get(f"{name}_val", container.get(name))
    return getattr(val, "values", val)

def reconstruct_wetness_recursive(precip, hl):
    phi = 2 ** (-1.0 / hl)
    w = np.zeros_like(precip, dtype=float)
    for t in range(1, len(precip)):
        w[t] = phi * w[t - 1] + (1 - phi) * precip[t]
    return w

def get_all_decompositions(idata, viz_meta, B_diel, precip_daily):
    """
    Vectorizes reconstruction for EVERY day in the sim timeline (110 days),
    not just days with audio windows.
    """
    post = idata.posterior
    n_draws = post.sizes["chain"] * post.sizes["draw"]
    n_days = len(viz_meta["full_calendar"])
    
    # Components on the DAILY scale
    decomp = {
        "beta_0": np.zeros(n_draws),
        "rain_fast": np.zeros((n_draws, n_days)),
        "rain_slow": np.zeros((n_draws, n_days)),
        "alpha_day": np.zeros((n_draws, n_days)),
        "diel_peak_val": np.zeros(n_draws), # The stationary peak of the diel spline
    }

    print(f"⚡ Reconstructing full 110-day timeline for {n_draws} draws...")
    for i, draw in enumerate(iterate_draws(idata)):
        # 1. Daily Components
        wf = reconstruct_wetness_recursive(precip_daily, get_param(draw, "half_life_fast"))
        ws = reconstruct_wetness_recursive(precip_daily, get_param(draw, "half_life_slow"))
        
        wf_std = wf / (np.std(wf) + 1e-6)
        decomp["rain_fast"][i] = get_param(draw, "gamma_fast") * (wf_std - np.mean(wf_std))
        
        ks = get_param(draw, "k_slow")
        ns = get_param(draw, "n_slow_val")

        ws_hill = (ws ** ns) / (ks ** ns + ws ** ns)

        decomp["rain_slow"][i] = (
            get_param(draw, "gamma_slow") *
            (ws_hill - np.mean(ws_hill))
        )
        
        decomp["alpha_day"][i] = get_param(draw, "alpha_day_raw") * get_param(draw, "sigma_day")
        decomp["beta_0"][i] = get_param(draw, "beta_0")
        
        # 2. Diel Component (Identify the peak value once per draw)
        beta_diel = np.concatenate([[0.0], np.cumsum(get_param(draw, "z_diel_raw") * get_param(draw, "sigma_diel"))])
        decomp["diel_peak_val"][i] = np.max(B_diel @ beta_diel)

    return decomp

# ============================================================
# Plotting Functions
# ============================================================

def plot_additive_component_synthesis(decomp, viz_meta, output_dir):
    """
    Plots the 5 additive components across the full 110-day calendar.
    Every day in September gets a value.
    """
    days = viz_meta["full_calendar"]
    n_days = len(days)
    
    plt.figure(figsize=(15, 8))
    colors = {"beta_0": "gray", "rain_fast": "blue", "rain_slow": "cyan", 
              "alpha_day": "purple", "diel_peak_val": "orange"}
    
    total_log_lambda = np.zeros_like(decomp["alpha_day"])
    
    for c, color in colors.items():
        vals = decomp[c]
        if len(vals.shape) == 1: # Broadcast beta_0 and diel_peak
            vals = np.tile(vals[:, None], (1, n_days))
        
        total_log_lambda += vals
        mu, lo, hi = np.nanmedian(vals, 0), np.nanpercentile(vals, 2.5, 0), np.nanpercentile(vals, 97.5, 0)
        plt.errorbar(days, mu, yerr=[mu-lo, hi-mu], fmt="o", color=color, 
                     label=c.title(), linewidth=0, elinewidth=1, alpha=0.5, markersize=3)

    # Plot Total
    mu_t, lo_t, hi_t = np.nanmedian(total_log_lambda, 0), np.nanpercentile(total_log_lambda, 2.5, 0), np.nanpercentile(total_log_lambda, 97.5, 0)
    plt.errorbar(days, mu_t, yerr=[mu_t-lo_t, hi_t-mu_t], fmt="o", color="black", 
                 label="Total Influence", linewidth=0, elinewidth=1.5, alpha=0.8, zorder=10, markersize=4)

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("Log-Intensity Decomposition (Full Timeline)")
    plt.ylabel("Log-scale Contribution")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "additive_synthesis_plot.png", dpi=150); plt.close()

def plot_seasonal_intensity_at_diel_peak(decomp, viz_meta, output_dir):
    """Intensity plot across all 110 days, regardless of audio availability."""
    days = viz_meta["full_calendar"]
    # Sum all log components: beta_0 + rf + rs + ad + diel_peak
    total_log = (decomp["beta_0"][:, None] + decomp["rain_fast"] + 
                 decomp["rain_slow"] + decomp["alpha_day"] + decomp["diel_peak_val"][:, None])
    
    lam = np.exp(total_log)
    mu, lo, hi = np.nanmedian(lam, 0), np.nanpercentile(lam, 2.5, 0), np.nanpercentile(lam, 97.5, 0)
    
    plt.figure(figsize=(14, 7))
    plt.errorbar(days, mu, yerr=[mu-lo, hi-mu], fmt="o", color="darkcyan", linewidth=0, elinewidth=1.2, alpha=0.6)
    plt.yscale("log")
    plt.title("Seasonal Intensity Trend (Full 110-Day Prediction)")
    plt.ylabel(r"Intensity $\lambda$ (Log Scale)")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(Path(output_dir) / "seasonal_peak_intensity_log.png"); plt.close()

def plot_day_random_effects(idata, viz_meta, output_dir):
    alpha = (idata.posterior.alpha_day_raw * idata.posterior.sigma_day).values
    flat = alpha.reshape(-1, alpha.shape[-1])
    mu, lo, hi = np.median(flat, 0), np.percentile(flat, 2.5, 0), np.percentile(flat, 97.5, 0)
    plt.figure(figsize=(14, 5)); plt.axhline(0, color='black', ls='--')
    plt.fill_between(viz_meta["full_calendar"], lo, hi, color='slategray', alpha=0.3)
    plt.plot(viz_meta["full_calendar"], mu, color='black', linewidth=1)
    plt.title(r"Daily Random Effects ($\alpha_{day}$)")
    plt.savefig(Path(output_dir) / "daily_random_effects.png"); plt.close()

def plot_learned_diel_cycle(idata, B_diel, windows_df, output_dir):
    sigma_diel = idata.posterior.sigma_diel.values
    z_diel_raw = idata.posterior.z_diel_raw.values
    beta = np.concatenate([np.zeros((*sigma_diel.shape, 1)), np.cumsum(z_diel_raw * sigma_diel[..., np.newaxis], axis=-1)], axis=-1)
    u_h, u_idx = np.unique(windows_df["mid_time_hour"].values, return_index=True)
    trend = np.tensordot(beta, B_diel[u_idx], axes=([-1], [1]))
    flat_trend = trend.reshape(-1, len(u_h))
    mu, lo, hi = np.median(flat_trend, 0), np.percentile(flat_trend, 2.5, 0), np.percentile(flat_trend, 97.5, 0)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1.plot(u_h, mu, color='firebrick'); ax1.fill_between(u_h, lo, hi, color='firebrick', alpha=0.2); ax1.set_title("Learned Diel Cycle")
    ax2.plot(u_h, np.exp(mu), color='navy'); ax2.set_ylabel(r"Multiplicative Factor ($\exp$)")
    plt.savefig(Path(output_dir) / "learned_diel_cycle.png", dpi=150); plt.close()

def plot_dual_rainfall_decay(idata, precip_daily, output_dir):
    """Plots daily-scale trajectories with saturation lines."""
    wf_l, ws_l, sat_l = [], [], []
    for d in iterate_draws(idata):
        wf = reconstruct_wetness_recursive(precip_daily, get_param(d, "half_life_fast"))
        ws = reconstruct_wetness_recursive(precip_daily, get_param(d, "half_life_slow"))
        ks = get_param(d, "k_slow")
        ns = get_param(d, "n_slow")

        sat = (ws ** ns) / (ks ** ns + ws ** ns)
        wf_l.append(wf); ws_l.append(ws); sat_l.append(sat)
    
    days = np.arange(len(precip_daily))
    fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Fast Scale
    ax[0].bar(days, precip_daily, alpha=0.15, color='blue', label="Precip")
    axt0 = ax[0].twinx()
    arr_f = np.array(wf_l)
    axt0.plot(days, arr_f.mean(0), color='tab:blue', label="Fast Wetness ($W_f$)")
    axt0.fill_between(days, np.percentile(arr_f, 2.5, 0), np.percentile(arr_f, 97.5, 0), color='tab:blue', alpha=0.2)
    ax[0].set_title("Fast Hydrological Scale (Linear)")

    # Slow Scale
    ax[1].bar(days, precip_daily, alpha=0.15, color='blue')
    axt1 = ax[1].twinx()
    arr_s, arr_sat = np.array(ws_l), np.array(sat_l)
    axt1.plot(days, arr_s.mean(0), color='tab:cyan', ls='--', alpha=0.6, label="Raw Slow ($W_s$)")
    axt1.plot(days, arr_sat.mean(0), color='tab:red', lw=2, label="Hill Effect ($Hill(W_s)$)")
    axt1.fill_between(days, np.percentile(arr_sat, 2.5, 0), np.percentile(arr_sat, 97.5, 0), color='tab:red', alpha=0.2)
    ax[1].set_title("Slow Hydrological Scale (Saturation)")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hydrological_dual_decay.png", dpi=150); plt.close()

def plot_hydrological_half_life(idata, output_dir):
    plt.figure(figsize=(10, 6))
    az.plot_kde(get_param(idata.posterior, "half_life_fast").flatten(), label="Fast")
    az.plot_kde(get_param(idata.posterior, "half_life_slow").flatten(), label="Slow")
    plt.legend(); plt.savefig(Path(output_dir) / "hydrological_half_life_kde.png"); plt.close()

def plot_mcmc_health(idata, output_dir):
    vars = [v for v in idata.posterior.data_vars if len(idata.posterior[v].dims) <= 3]
    az.plot_rank(idata, var_names=vars, kind="vlines"); plt.savefig(Path(output_dir) / "rank_plot.png"); plt.close()
    az.summary(idata, var_names=vars).to_csv(Path(output_dir) / "mcmc_summary.csv")

def iterate_draws(idata):
    post = idata.posterior
    for c in range(post.sizes["chain"]):
        for d in range(post.sizes["draw"]):
            yield {k: post[k].values[c, d] for k in post.data_vars}

def plot_total_rain_influence(idata, viz_meta, precip_daily, output_dir):
    all_t = []
    for draw in iterate_draws(idata):
        wf = reconstruct_wetness_recursive(precip_daily, get_param(draw, "half_life_fast"))
        ws = reconstruct_wetness_recursive(precip_daily, get_param(draw, "half_life_slow"))
        r_f = get_param(draw, "gamma_fast") * (wf/(np.std(wf)+1e-6) - np.mean(wf/(np.std(wf)+1e-6)))
        r_s = get_param(draw, "gamma_slow") * (
            ws/(get_param(draw, "k_slow")+ws) 
            - np.mean(ws/(get_param(draw, "k_slow")+ws))
        )
    all_t.append(r_f + r_s)
    mu, lo, hi = np.mean(all_t, 0), np.percentile(all_t, 2.5, 0), np.percentile(all_t, 97.5, 0)
    plt.figure(figsize=(14, 6)); plt.plot(viz_meta["full_calendar"], mu)
    plt.fill_between(viz_meta["full_calendar"], lo, hi, alpha=0.2)
    plt.title("Total Rain Influence (Full Timeline)"); plt.savefig(Path(output_dir) / "total_rain_influence.png"); plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/breallis/dev/frog_perch/stat_results/call_intensity_spline_rainfall_v8"),
        help="Directory containing inference outputs (default: current directory)"
    )
    args = parser.parse_args()

    idata = az.from_netcdf(args.output_dir / "inference_data_rain.nc")
    windows_df = pd.read_csv(args.output_dir / "windowed_detector_data.csv")
    m_params = np.load(args.output_dir / "model_params.npz")
    
    viz_meta = preprocess_viz_metadata(windows_df, m_params["precip_daily"], int(m_params["burn_in_days"]))
    decomp = get_all_decompositions(idata, viz_meta, m_params["B_diel"], m_params["precip_daily"])

    plot_seasonal_intensity_at_diel_peak(decomp, viz_meta, args.output_dir)
    plot_additive_component_synthesis(decomp, viz_meta, args.output_dir)
    plot_mcmc_health(idata, args.output_dir)
    plot_hydrological_half_life(idata, args.output_dir)
    plot_dual_rainfall_decay(idata, m_params["precip_daily"], args.output_dir)
    plot_learned_diel_cycle(idata, m_params["B_diel"], windows_df, args.output_dir)
    plot_day_random_effects(idata, viz_meta, args.output_dir)
    plot_total_rain_influence(idata, viz_meta, m_params["precip_daily"], args.output_dir)
    
    print("✅ Full Optimized Visualization Suite complete.")

if __name__ == "__main__":
    main()