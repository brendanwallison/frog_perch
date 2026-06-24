#!/usr/bin/env python3
"""
Memory-Safe, High-Performance Visualization for the Clean 3-Day Linear Lag Matrix Model.
Universal source of truth for structural cross-scale decomposition and temporal alignment.
Updated for Dynamic B-Splines and Unified, Continuous Global Climate Architecture.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import jax
import jax.numpy as jnp

plt.style.use("ggplot")

# ============================================================
# Spline Reconstruction Utility
# ============================================================

def jax_b3_spline_basis(x, knots, low_bound=16.0, up_bound=24.0):
    """Reconstructs the clamped basis matrix dynamically for visualization."""
    all_knots = jnp.concatenate([
        jnp.repeat(low_bound, 4), 
        knots, 
        jnp.repeat(up_bound, 4)
    ])
    x_clip = jnp.clip(x, low_bound, up_bound)
    
    n_knots = len(all_knots)
    n_bases = n_knots - 1
    
    is_last = (x_clip[:, None] >= all_knots[-2]) & (x_clip[:, None] <= all_knots[-1])
    is_others = (x_clip[:, None] >= all_knots[:-1]) & (x_clip[:, None] < all_knots[1:])
    
    mask = jnp.where(is_last, True, is_others)
    B = [jnp.where(mask, 1.0, 0.0)]
    
    for deg in range(1, 4):
        B_next = []
        for i in range(n_bases - deg):
            denom1 = all_knots[i + deg] - all_knots[i]
            denom2 = all_knots[i + deg + 1] - all_knots[i + 1]
            
            d1_safe = jnp.where(denom1 > 0, denom1, 1.0)
            d2_safe = jnp.where(denom2 > 0, denom2, 1.0)
            
            term1 = jnp.where(denom1 > 0, 1.0, 0.0) * ((x_clip - all_knots[i]) / d1_safe) * B[-1][:, i]
            term2 = jnp.where(denom2 > 0, 1.0, 0.0) * ((all_knots[i + deg + 1] - x_clip) / d2_safe) * B[-1][:, i + 1]
            
            B_next.append((term1 + term2)[:, None])
            
        B.append(jnp.hstack(B_next))
        
    return np.array(B[-1]) # Convert back to standard NumPy for Matplotlib

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

def get_param(container, name):
    val = container.get(f"{name}_val", container.get(name))
    return getattr(val, "values", val)

def reconstruct_slow_wetness(precip, hl):
    phi = 2 ** (-1.0 / hl)
    w = np.zeros_like(precip, dtype=float)
    for t in range(1, len(precip)):
        w[t] = phi * w[t - 1] + (1 - phi) * precip[t]
    return w

def iterate_draws(idata):
    post = idata.posterior
    for c in range(post.sizes["chain"]):
        for d in range(post.sizes["draw"]):
            return_dict = {}
            for k in post.data_vars:
                # Force extract deterministic tracked values if they exist
                if f"{k}_val" in post.data_vars:
                    return_dict[k] = post[f"{k}_val"].values[c, d]
                else:
                    return_dict[k] = post[k].values[c, d]
            yield return_dict

def get_all_decompositions(idata, viz_meta, B_diel, m_params):
    post = idata.posterior
    n_draws = post.sizes["chain"] * post.sizes["draw"]
    precip_daily = m_params["precip_daily"]
    n_days = len(precip_daily)
    
    temp = m_params["temp"]
    rh = m_params["rh"]
    rms_obs = m_params.get("rms_obs", idata.constant_data["rms_obs"].values if "rms_obs" in idata.constant_data else None)
    
    day_idx = viz_meta["day_idx"]

    decomp = {
        "beta_0": np.zeros(n_draws),
        "fast_rain_raw": np.zeros((n_draws, n_days)),
        "eff_slow_raw": np.zeros((n_draws, n_days)),
        "eff_interact": np.zeros((n_draws, n_days)),
        "alpha_day": np.zeros((n_draws, n_days)),
        "diel_peak_val": np.zeros(n_draws), 
        "eff_climate_5s": np.zeros((n_draws, len(temp)))
    }

    p0 = precip_daily
    p1 = np.concatenate([[0.0], precip_daily[:-1]])
    p2 = np.concatenate([[0.0, 0.0], precip_daily[:-2]])
    
    print(f"⚡ Decompiling multi-scale features for {n_draws} posteriors...")
    for i, draw in enumerate(iterate_draws(idata)):
        b_p0 = draw["b_p0"]
        b_p1 = draw["b_p1"]
        b_p2 = draw["b_p2"]
        
        fast_rain = (b_p0 * p0) + (b_p1 * p1) + (b_p2 * p2)
        decomp["fast_rain_raw"][i] = fast_rain

        ws = reconstruct_slow_wetness(precip_daily, draw["half_life_slow"])
        tau = draw["tau_pool"]
        b_shape = draw["b_shape"]
        trigger = 1.0 / (1.0 + np.exp(-b_shape * (ws - tau)))
        
        gamma_plateau = draw["gamma_plateau"]
        decomp["eff_slow_raw"][i] = gamma_plateau * trigger
        decomp["eff_interact"][i] = 0.0 # Interaction terms safely zeroed out for simpler graph
        
        decomp["alpha_day"][i] = draw["alpha_day_raw"] * draw["b_day"]
        decomp["beta_0"][i] = draw["beta_0"]
        
        beta_diel = np.concatenate([[0.0], np.cumsum(draw["z_diel_raw"] * draw["sigma_diel"])])
        decomp["diel_peak_val"][i] = np.max(B_diel @ beta_diel)

        decomp["eff_climate_5s"][i] = (
            (draw["b_temp"] * temp) + 
            (draw["b_rh"] * rh) + 
            (draw["b_rms"] * rms_obs)
        )

    return decomp

# ============================================================
# Plotting Suite
# ============================================================

def plot_additive_component_synthesis(decomp, viz_meta, output_dir):
    days = viz_meta["full_calendar"]
    n_days = len(days)
    
    plt.figure(figsize=(15, 8), dpi=150)
    
    b0 = decomp["beta_0"][:, None]
    diel = decomp["diel_peak_val"][:, None]
    fast = decomp["fast_rain_raw"]
    slow = decomp["eff_slow_raw"]
    alpha = decomp["alpha_day"]
    
    base_scalars = b0 + diel
    stack_base = np.broadcast_to(base_scalars, (base_scalars.shape[0], n_days))
    
    stack_with_slow = stack_base + slow
    stack_with_weather = stack_with_slow + fast
    stack_total = stack_with_weather + alpha
    
    def get_stats(stack_matrix):
        return (np.nanmedian(stack_matrix, axis=0), 
                np.nanpercentile(stack_matrix, 2.5, axis=0), 
                np.nanpercentile(stack_matrix, 97.5, axis=0))
    
    mu_base, lo_base, hi_base = get_stats(stack_base)
    mu_slow, lo_slow, hi_slow = get_stats(stack_with_slow)
    mu_weath, lo_weath, hi_weath = get_stats(stack_with_weather)
    mu_tot, lo_tot, hi_tot = get_stats(stack_total)
    
    plt.plot(days, mu_base, color="gray", linestyle=":", label="Diel Peak Baseline", lw=1.5)
    plt.fill_between(days, lo_base, hi_base, color="gray", alpha=0.08)
    
    plt.plot(days, mu_slow, color="cyan", label="Base + Eff Slow (Plateau Capacity)", lw=2)
    plt.fill_between(days, mu_base, hi_slow, color="cyan", alpha=0.15)
    
    plt.plot(days, mu_weath, color="blue", label="Base + Slow + Fast Rain Response", lw=2)
    plt.fill_between(days, mu_slow, hi_weath, color="blue", alpha=0.2)
    
    plt.plot(days, mu_tot, color="black", label="Total Composite Influence (Includes Alpha Day)", lw=2.5, zorder=10)
    plt.fill_between(days, lo_tot, hi_tot, color="purple", alpha=0.12, label="Alpha Day Unmodeled Bounds", zorder=1)

    plt.axhline(0, color='black', linestyle='--', alpha=0.3)
    plt.title("Log-Intensity Timeline Decomposition (Cumulative Unified Model Stack)", fontsize=14, pad=15)
    plt.ylabel("Cumulative Log-scale Contribution", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor="white", framealpha=0.9)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "additive_synthesis_plot.png", dpi=150)
    plt.close()

def plot_seasonal_intensity_at_diel_peak_abs(decomp, viz_meta, output_dir):
    days = viz_meta["full_calendar"]
    total_log = (decomp["beta_0"][:, None] + decomp["fast_rain_raw"] + 
                 decomp["eff_slow_raw"] + decomp["alpha_day"] + decomp["diel_peak_val"][:, None])
    
    log_offset = np.log(60.0 / 5.0) 
    log_rate = log_offset + total_log
    lam = np.exp(log_rate)
    
    mu, lo, hi = np.nanmedian(lam, 0), np.nanpercentile(lam, 2.5, 0), np.nanpercentile(lam, 97.5, 0)
    
    plt.figure(figsize=(14, 7))
    plt.errorbar(days, mu, yerr=[mu-lo, hi-mu], fmt="o", color="darkcyan", linewidth=0, elinewidth=1.2, alpha=0.6)
    plt.title("Seasonal Call Intensity Target Tracking (Absolute Window Scale)")
    plt.ylabel(r"Expected Event Rate $\lambda$ (calls / minute)")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(Path(output_dir) / "seasonal_peak_intensity_abs.png"); plt.close()

def plot_seasonal_intensity_at_diel_peak(decomp, viz_meta, output_dir):
    days = viz_meta["full_calendar"]
    total_log = (decomp["beta_0"][:, None] + decomp["fast_rain_raw"] + 
                 decomp["eff_slow_raw"] + decomp["alpha_day"] + decomp["diel_peak_val"][:, None])
    
    log_offset = np.log(60.0 / 5.0) 
    log_rate = log_offset + total_log
    lam = np.exp(log_rate)
    
    mu, lo, hi = np.nanmedian(lam, 0), np.nanpercentile(lam, 2.5, 0), np.nanpercentile(lam, 97.5, 0)
    
    plt.figure(figsize=(14, 7))
    plt.errorbar(days, mu, yerr=[mu-lo, hi-mu], fmt="o", color="darkcyan", linewidth=0, elinewidth=1.2, alpha=0.6)
    plt.yscale("log")
    plt.title("Seasonal Call Intensity Target Tracking (Multiplicative Scale)")
    plt.ylabel(r"Intensity $\lambda$ (Log Scale)")
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(Path(output_dir) / "seasonal_peak_intensity_log.png"); plt.close()

def plot_absolute_scale_multipliers(decomp, viz_meta, output_dir):
    days = viz_meta["full_calendar"]
    
    plt.figure(figsize=(15, 8), dpi=150)
    mult_slow = np.exp(decomp["eff_slow_raw"])
    mult_fast = np.exp(decomp["fast_rain_raw"])
    mult_alpha = np.exp(decomp["alpha_day"])
    
    def get_mult_stats(matrix):
        return (np.nanmedian(matrix, axis=0), 
                np.nanpercentile(matrix, 2.5, axis=0), 
                np.nanpercentile(matrix, 97.5, axis=0))
        
    mu_slow, lo_slow, hi_slow = get_mult_stats(mult_slow)
    mu_fast, lo_fast, hi_fast = get_mult_stats(mult_fast)
    mu_alpha, lo_alpha, hi_alpha = get_mult_stats(mult_alpha)
    
    plt.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Baseline (No Weather Impact)")
    
    plt.plot(days, mu_slow, color="cyan", label="Seasonal Multiplicative Lift (Eff Slow)", lw=2)
    plt.fill_between(days, lo_slow, hi_slow, color="cyan", alpha=0.15)
    
    plt.plot(days, mu_fast, color="blue", label="Short-term Storm Pulse Factor (Fast Rain)", lw=1.5, alpha=0.8)
    plt.fill_between(days, lo_fast, hi_fast, color="blue", alpha=0.12)
    
    plt.plot(days, mu_alpha, color="purple", label="Daily Unmodeled Variation Factor (Alpha Day)", lw=1, alpha=0.6)
    plt.fill_between(days, lo_alpha, hi_alpha, color="purple", alpha=0.08)
    
    plt.yscale("log")
    from matplotlib.ticker import FormatStrFormatter
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%g x'))
    
    plt.title("Absolute-Scale Multiplicative Impact Timeline (Decompiled Impact Factors)", fontsize=14, pad=15)
    plt.ylabel("Intensity Scaling Multiplier (Log-Distributed Scale)", fontsize=12)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), facecolor="white", framealpha=0.9)
    plt.xticks(rotation=45)
    plt.grid(True, which="both", ls="-", alpha=0.15)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "absolute_factors_timeline.png", dpi=150)
    plt.close()

def plot_day_random_effects(idata, viz_meta, output_dir):
    post = idata.posterior
    b_day = post.get("b_day_val", post.get("b_day"))
    alpha = (post.alpha_day_raw * b_day).values
    flat = alpha.reshape(-1, alpha.shape[-1])
    mu, lo, hi = np.median(flat, 0), np.percentile(flat, 2.5, 0), np.percentile(flat, 97.5, 0)
    plt.figure(figsize=(14, 5)); plt.axhline(0, color='black', ls='--')
    plt.fill_between(viz_meta["full_calendar"], lo, hi, color='slategray', alpha=0.3)
    plt.plot(viz_meta["full_calendar"], mu, color='black', linewidth=1)
    plt.title(r"Daily Baseline Fluctuation Residuals ($\alpha_{day}$)")
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
    ax1.plot(u_h, mu, color='firebrick'); ax1.fill_between(u_h, lo, hi, color='firebrick', alpha=0.2); ax1.set_title("Learned Circadian Diel Curve")
    ax2.plot(u_h, np.exp(mu), color='navy'); ax2.set_ylabel(r"Circadian Scale Multiplier ($\exp$)")
    plt.savefig(Path(output_dir) / "learned_diel_cycle.png", dpi=150); plt.close()

def plot_structural_hydrology_and_lags(idata, precip_daily, output_dir):
    from scipy.stats import gaussian_kde
    
    trigger_l = []
    for d in iterate_draws(idata):
        ws = reconstruct_slow_wetness(precip_daily, d["half_life_slow"])
        tau = d["tau_pool"]
        b_shape = d["b_shape"]
        trigger = 1.0 / (1.0 + np.exp(-b_shape * (ws - tau)))
        trigger_l.append(trigger)
    
    days = np.arange(len(precip_daily))
    fig, ax = plt.subplots(2, 1, figsize=(12, 10))
    
    ax[0].bar(days, precip_daily, alpha=0.15, color='blue')
    axt0 = ax[0].twinx()
    axt0.plot(days, np.array(trigger_l).mean(0), color='tab:green', lw=2, label="Seasonal Carrying Capacity Gate (0 to 1)")
    ax[0].set_title("Slow Landscape Hydrology Switch State")
    axt0.legend(loc='upper left')

    post = idata.posterior
    lag_names = ["b_p0 (Day-Of)", "b_p1 (Lag 1)", "b_p2 (Lag 2)"]
    lag_vars = [post["b_p0"].values.flatten(), post["b_p1"].values.flatten(), post["b_p2"].values.flatten()]
    
    for val, name in zip(lag_vars, lag_names):
        kde = gaussian_kde(val)
        x_eval = np.linspace(val.min() - val.std(), val.max() + val.std(), 200)
        ax[1].plot(x_eval, kde(x_eval), label=name, lw=2)
        
    ax[1].axvline(0.0, color='black', ls='--', alpha=0.5)
    ax[1].set_title("Posterior Marginals of Short-Term Flash Hydrology Matrix")
    ax[1].set_xlabel("Slope Value (Log Scale Effect Size)")
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hydrological_structural_lags.png", dpi=150)
    plt.close()

def plot_total_rain_influence(idata, viz_meta, precip_daily, output_dir):
    all_t = []
    p0 = precip_daily
    p1 = np.concatenate([[0.0], precip_daily[:-1]])
    p2 = np.concatenate([[0.0, 0.0], precip_daily[:-2]])
    
    for draw in iterate_draws(idata):
        fast_rain = (draw["b_p0"] * p0) + (draw["b_p1"] * p1) + (draw["b_p2"] * p2)
        ws = reconstruct_slow_wetness(precip_daily, draw["half_life_slow"])
        trigger = 1.0 / (1.0 + np.exp(-draw["b_shape"] * (ws - draw["tau_pool"])))
        r_s = draw["gamma_plateau"] * trigger
        all_t.append(fast_rain + r_s)
        
    mu, lo, hi = np.mean(all_t, 0), np.percentile(all_t, 2.5, 0), np.percentile(all_t, 97.5, 0)
    plt.figure(figsize=(14, 6)); plt.plot(viz_meta["full_calendar"], mu, color='dodgerblue')
    plt.fill_between(viz_meta["full_calendar"], lo, hi, alpha=0.2, color='dodgerblue')
    plt.title("Total Macro-Hydrological Core Influence Profile (Combined Basin Trackers)")
    plt.ylabel("Combined Log-scale Rain Impact")
    plt.savefig(Path(output_dir) / "total_rain_influence.png"); plt.close()

def plot_mcmc_health(idata, output_dir):
    priority_vars = [
        "beta_0", "phi", "b_p0", "b_p1", "b_p2", 
        "half_life_slow_val", "tau_pool_val", "b_shape_val",
        "gamma_plateau_val", "b_rms_val",
        "b_temp_val", "b_rh_val", 
        "sigma_diel", "b_day_val", "delta_seasonal_val"
    ]
    
    plot_vars = [v for v in priority_vars if v in idata.posterior.data_vars or f"{v}_val" in idata.posterior.data_vars]
    
    print("📊 Generating MCMC health rank plots for core parameters...")
    az.plot_rank(idata, var_names=plot_vars)
    plt.savefig(Path(output_dir) / "rank_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print("💾 Saving detailed MCMC summary CSV...")
    all_vars = [v for v in idata.posterior.data_vars]
    az.summary(idata, var_names=all_vars).to_csv(Path(output_dir) / "mcmc_summary.csv")

def plot_microclimate_marginals(idata, output_dir):
    """Plots the unified, single-scale posterior slopes for the continuous model."""
    from scipy.stats import gaussian_kde
    
    plt.figure(figsize=(12, 6), dpi=150)
    post = idata.posterior
    
    climate_vars = {
        r"Global Temperature ($b_{temp}$)": post["b_temp"].values.flatten(),
        r"Global Relative Humidity ($b_{rh}$)": post["b_rh"].values.flatten(),
        r"Acoustic Noise Suppression ($b_{rms}$)": post["b_rms"].values.flatten()
    }
    
    colors = ["#e74c3c", "#3498db", "#2c3e50"]
    for (name, chain_draws), color in zip(climate_vars.items(), colors):
        kde = gaussian_kde(chain_draws)
        x_eval = np.linspace(chain_draws.min() - chain_draws.std(), chain_draws.max() + chain_draws.std(), 200)
        plt.plot(x_eval, kde(x_eval), label=name, color=color, lw=2)
        
    plt.axvline(0.0, color='black', linestyle='--', alpha=0.6)
    plt.title("Unified Single-Scale Environmental Posterior Slopes", fontsize=13, pad=15)
    plt.xlabel("Standardized Log-scale Effect Size (β)", fontsize=11)
    plt.ylabel("Posterior Density", fontsize=11)
    plt.legend(facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "microclimate_kinetic_marginals.png", dpi=150)
    plt.close()

def plot_multiscale_phase_portrait(decomp, idata, viz_meta, output_dir):
    plt.figure(figsize=(11, 8), dpi=150)
    
    mean_slow_gate = np.nanmedian(decomp["eff_slow_raw"], axis=0) 
    mean_climate_5s = np.nanmedian(decomp["eff_climate_5s"], axis=0) 
    
    day_idx = viz_meta["day_idx"]
    mapped_gate_5s = mean_slow_gate[day_idx]
    
    total_log_rate = np.nanmedian(
        decomp["beta_0"][:, None] + decomp["eff_climate_5s"] + 
        (decomp["fast_rain_raw"] + decomp["eff_slow_raw"])[:, day_idx] + 
        decomp["alpha_day"][:, day_idx], axis=0
    )
    
    sc = plt.scatter(
        mapped_gate_5s, 
        mean_climate_5s, 
        c=total_log_rate, 
        cmap="viridis", 
        s=12, 
        alpha=0.4, 
        edgecolors="none"
    )
    
    cbar = plt.colorbar(sc)
    cbar.set_label("Net Expected Call Rate (Log Scale)", fontsize=11)
    
    plt.title("Unified Behavioral Phase Portrait (State-Space Coordination)", fontsize=13, pad=15)
    plt.xlabel("Macro-Scale Carrying Capacity Switch State (Slow Hydrology Trigger)", fontsize=11)
    plt.ylabel("High-Frequency Environmental & Noise Pressure (Sub-Daily Score)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "multiscale_phase_portrait.png", dpi=150)
    plt.close()

def plot_rainfall_marginals(idata, output_dir):
    from scipy.stats import gaussian_kde
    
    plt.figure(figsize=(11, 6), dpi=150)
    post = idata.posterior
    
    rainfall_vars = {
        r"Day-Of Pulse ($b_{p0}$)": post["b_p0"].values.flatten(),
        r"1-Day Lag Recovery ($b_{p1}$)": post["b_p1"].values.flatten(),
        r"2-Day Lag Recovery ($b_{p2}$)": post["b_p2"].values.flatten(),
        r"Seasonal Plateau Amp ($\gamma_{plateau}$)": post["gamma_plateau"].values.flatten()
    }
    
    colors = ["#2ecc71", "#27ae60", "#1abc9c", "#9b59b6"]
    for (name, chain_draws), color in zip(rainfall_vars.items(), colors):
        kde = gaussian_kde(chain_draws)
        x_eval = np.linspace(chain_draws.min() - chain_draws.std(), chain_draws.max() + chain_draws.std(), 200)
        plt.plot(x_eval, kde(x_eval), label=name, color=color, lw=2)
        
    plt.axvline(0.0, color='black', linestyle='--', alpha=0.6)
    plt.title("Posterior Marginals of Hydrological Process Weights", fontsize=13, pad=15)
    plt.xlabel("Log-scale Parameter Value (Effect Size Impact)", fontsize=11)
    plt.ylabel("Posterior Density", fontsize=11)
    plt.legend(facecolor="white", framealpha=0.9, loc="upper right")
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "hydrological_process_marginals.png", dpi=150)
    plt.close()

def plot_climate_timescale_decomposition(decomp, viz_meta, windows_df, output_dir):
    """Variance Partitioning check on unified climate predictors."""
    joint_climate_effect = np.nanmedian(decomp["eff_climate_5s"], axis=0)
    
    df_calc = pd.DataFrame({
        "datetime": pd.to_datetime(windows_df["start_time"]),
        "hour": windows_df["mid_time_hour"],
        "signal": joint_climate_effect
    })
    df_calc["date"] = df_calc["datetime"].dt.date
    df_calc["minute_block"] = df_calc["datetime"].dt.floor("1Min").dt.time

    daily_means = df_calc.groupby("date")["signal"].transform("mean")
    df_calc["daily_resid"] = df_calc["signal"] - daily_means
    diel_profile = df_calc.groupby("minute_block")["daily_resid"].transform("mean")
    residual_variance = df_calc["daily_resid"] - diel_profile

    var_daily = np.var(daily_means.values)
    var_diel = np.var(diel_profile.values)
    var_resid = np.var(residual_variance.values)

    sum_of_vars = var_daily + var_diel + var_resid

    print("\n🧮 --- Unified Hierarchical Variance Partitioning ---")
    print(f" Total Reconstructed Variance (Sample Sum):  {sum_of_vars:.6f}")
    print(f" ──► Daily Baseline Component Variance:      {var_daily:.6f} ({var_daily/sum_of_vars*100:.1f}%)")
    print(f" ──► Diel Circadian Component Variance:      {var_diel:.6f} ({var_diel/sum_of_vars*100:.1f}%)")
    print(f" ──► Window Residual Scale Variance:         {var_resid:.6f} ({var_resid/sum_of_vars*100:.1f}%)")
    print("----------------------------------------------------\n")

    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=False, dpi=150)
    
    unique_dates = df_calc.groupby("date")["datetime"].first()
    unique_daily_vals = df_calc.groupby("date")["signal"].mean()
    axes[0].plot(unique_dates, unique_daily_vals, color="#2ecc71", marker="o", lw=2)
    axes[0].set_title(f"Macro-Scale Daily Variance Component (Contribution: {var_daily/sum_of_vars*100:.1f}%)", fontsize=11)
    axes[0].set_ylabel("Log-scale Effect Mean")
    axes[0].tick_params(axis='x', rotation=25)

    u_hours, u_idx = np.unique(df_calc["hour"].values, return_index=True)
    sorted_diel_vals = diel_profile.values[u_idx]
    sort_sort = np.argsort(u_hours)
    axes[1].plot(u_hours[sort_sort], sorted_diel_vals[sort_sort], color="#3498db", lw=2.5)
    axes[1].set_title(f"Sub-Daily Diel Circadian Component Component (Contribution: {var_diel/sum_of_vars*100:.1f}%)", fontsize=11)
    axes[1].set_xlabel("Time of Day (Hours)")
    axes[1].set_ylabel("Log-scale Effect Profile")

    axes[2].hist(residual_variance, bins=60, density=True, color="#9b59b6", alpha=0.75, edgecolor="none")
    axes[2].axvline(0.0, color="black", linestyle="--", alpha=0.6)
    axes[2].set_title(f"High-Frequency Window Residual Scale Distribution (Contribution: {var_resid/sum_of_vars*100:.1f}%)", fontsize=11)
    axes[2].set_xlabel("Residual Log-scale Deviation Shocks")
    axes[2].set_ylabel("Density")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "climate_timescale_analysis.png", dpi=150)
    plt.close()

# ============================================================
# Gap-Aware Continuous Environmental Heatmaps
# ============================================================

def plot_comprehensive_effect_heatmaps(idata, m_params, windows_df, decomp, viz_meta, B_diel, output_dir):
    """Generates heatmaps parsing unified inputs into explicit, gap-aware log-scale parameters."""
    print("🌡️ Generating Comprehensive Unified Effect Heatmaps...")
    post = idata.posterior

    # --- 1. Extract Median Slopes ---
    b_temp = float(post["b_temp"].median())
    b_rh = float(post["b_rh"].median())

    # --- 2. Compute Effect Heatmap Inputs (X * Beta) ---
    eff_temp = b_temp * m_params["temp"]
    eff_rh = b_rh * m_params["rh"]

    # --- 3. Compute Total Unified Model Log-Rate (Lambda) ---
    b0 = float(post["beta_0"].median())
    fast_rain = np.median(decomp["fast_rain_raw"], axis=0)
    slow_rain = np.median(decomp["eff_slow_raw"], axis=0)
    daily_stack = fast_rain + slow_rain
    day_idx = viz_meta["day_idx"]

    # Reconstruct median biological diel curve
    sigma_diel = post.sigma_diel.values
    z_diel_raw = post.z_diel_raw.values
    beta_diel_med = np.median(np.concatenate([np.zeros((*sigma_diel.shape, 1)), np.cumsum(z_diel_raw * sigma_diel[..., np.newaxis], axis=-1)], axis=-1), axis=(0,1))
    trend_diel = B_diel @ beta_diel_med

    # Climate stack natively holds standard error thresholds 
    climate_stack = np.median(decomp["eff_climate_5s"], axis=0)
    total_log_rate = b0 + daily_stack[day_idx] + trend_diel + climate_stack

    # --- 4. Build Alignment Grid DataFrame ---
    df = pd.DataFrame({
        "datetime": pd.to_datetime(windows_df["start_time"]),
        "eff_temp": eff_temp,
        "eff_rh": eff_rh,
        "total_log_rate": total_log_rate
    })
    df["date"] = df["datetime"].dt.date
    df["time_bin"] = df["datetime"].dt.floor("10min").dt.time 

    # --- 5. Plotting Rendering Wrapper ---
    def draw_heatmap_row(cols, titles, cmaps, output_name, super_title, vmin=None, vmax=None):
        n_plots = len(cols)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 10), sharey=True, dpi=150)
        if n_plots == 1: axes = [axes]

        for i, (col, title, cmap) in enumerate(zip(cols, titles, cmaps)):
            # Pivoting naturally leaves audio recording gaps blank (NaN/Gray background)
            pivot = df.pivot_table(index="date", columns="time_bin", values=col, aggfunc="mean")

            if vmin is None and vmax is None:
                abs_max = np.nanmax(np.abs(pivot.values))
                c_vmin, c_vmax = -abs_max, abs_max
            else:
                c_vmin, c_vmax = vmin, vmax

            axes[i].set_facecolor('#e0e0e0') # Gray mask for unrecorded nights
            im = axes[i].imshow(pivot.values, aspect="auto", cmap=cmap, vmin=c_vmin, vmax=c_vmax, interpolation="none")
            axes[i].set_title(title, fontsize=13, pad=10)

            y_ticks = np.arange(0, len(pivot.index), max(1, len(pivot.index)//15))
            axes[i].set_yticks(y_ticks)
            axes[i].set_yticklabels([pivot.index[j].strftime("%b %d") for j in y_ticks])

            x_ticks = np.arange(0, len(pivot.columns), max(1, len(pivot.columns)//5))
            axes[i].set_xticks(x_ticks)
            axes[i].set_xticklabels([pivot.columns[j].strftime("%H:%M") for j in x_ticks])
            axes[i].set_xlabel("Time of Day (17:00 - 23:00)", fontsize=11)

            cbar = fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
            if "total_log_rate" in col:
                cbar.set_label("Total Expected Call Rate (Log λ)", fontsize=11)
            else:
                cbar.set_label("Effect Size (Δ Log λ)", fontsize=11)

        plt.suptitle(super_title, fontsize=16, y=0.96)
        plt.tight_layout()
        plt.savefig(Path(output_dir) / output_name, dpi=150)
        plt.close()

    # --- Render Component Heatmaps ---
    cols1 = ["eff_temp", "eff_rh"]
    titles1 = ["Unified Temperature Effect", "Unified Humidity Effect"]
    cmaps1 = ["RdBu", "BrBG"]
    
    # Passing arguments exactly matched to the updated function signature
    draw_heatmap_row(cols1, titles1, cmaps1, "heatmap_effects_2_combined.png", "Continuous Parameter Environmental Effects (Log-Scale Weightings)")

    # --- Render Grand Mean Output Prediction ---
    draw_heatmap_row(["total_log_rate"], ["Total Reconstructed Call Intensity"], ["viridis"], 
                     "heatmap_effects_3_total_rate.png", "Total Comprehensive Model Output (Expected Log Rate)", 
                     vmin=df["total_log_rate"].min(), vmax=df["total_log_rate"].max())
    
    print("✅ Effect heatmaps saved successfully.")

# ============================================================
# Main Execution Loop
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/breallis/dev/frog_perch/stat_results/model_final"),
        help="Directory containing inference outputs"
    )
    args = parser.parse_args()

    idata = az.from_netcdf(args.output_dir / "inference_data_rain.nc")
    windows_df = pd.read_csv(args.output_dir / "windowed_detector_data.csv")
    m_params = np.load(args.output_dir / "model_params.npz")
    
    viz_meta = preprocess_viz_metadata(windows_df, m_params["precip_daily"], int(m_params["burn_in_days"]))
    B_diel = jax_b3_spline_basis(m_params["time_of_day"], m_params["knots_grid"])
    decomp = get_all_decompositions(idata, viz_meta, B_diel, m_params)

    print("🎨 Generating unified plot suite...")
    plot_absolute_scale_multipliers(decomp, viz_meta, args.output_dir)
    plot_seasonal_intensity_at_diel_peak(decomp, viz_meta, args.output_dir)
    plot_seasonal_intensity_at_diel_peak_abs(decomp, viz_meta, args.output_dir)
    plot_additive_component_synthesis(decomp, viz_meta, args.output_dir)
    plot_mcmc_health(idata, args.output_dir)
    plot_structural_hydrology_and_lags(idata, m_params["precip_daily"], args.output_dir)
    plot_learned_diel_cycle(idata, B_diel, windows_df, args.output_dir)
    plot_day_random_effects(idata, viz_meta, args.output_dir)
    plot_total_rain_influence(idata, viz_meta, m_params["precip_daily"], args.output_dir)

    plot_microclimate_marginals(idata, args.output_dir)
    plot_rainfall_marginals(idata, args.output_dir)
    plot_multiscale_phase_portrait(decomp, idata, viz_meta, args.output_dir)
    plot_climate_timescale_decomposition(decomp, viz_meta, windows_df, args.output_dir)
    
    plot_comprehensive_effect_heatmaps(idata, m_params, windows_df, decomp, viz_meta, B_diel, args.output_dir)
    
    print("✅ Complete visualization execution finished successfully.")

if __name__ == "__main__":
    main()