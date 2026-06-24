#!/usr/bin/env python3
"""
visualize_calibration_saturation.py

Generates publication-isolated figures for:
1) Standalone Weibull Saturation Curve
2) Whole-number aligned, clipped Beta Density Ridgeplot

Both plots maintain strict figure dimensions linked to axis scales.
"""

from __future__ import annotations

import json
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.stats import beta as beta_dist

import configs.nn_config as config

# ---------------------------------------------------------------------------
# Global Config & Aesthetics
# ---------------------------------------------------------------------------
TRUE_COUNT_COLUMN = "gt_mu"

from matplotlib import rcParams

def theme_custom():
    """Apply custom theme matching R ggplot style."""
    # Font settings
    rcParams['font.family'] = 'Times New Roman'
    rcParams['font.size'] = 16
   
    # Axes
    rcParams['axes.labelsize'] = 25  # rel(1.75) * base size
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.linewidth'] = 1
    rcParams['axes.labelpad'] = 12
   
    # Ticks
    rcParams['xtick.labelsize'] = 24  # rel(1.5) * base size
    rcParams['ytick.labelsize'] = 24
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
   
    # Grid
    rcParams['axes.grid'] = False
   
    # Legend
    rcParams['legend.fontsize'] = 18  # rel(1.1)
    rcParams['legend.title_fontsize'] = 20  # rel(1.2)
    rcParams['legend.frameon'] = True
    rcParams['legend.fancybox'] = False
    rcParams['legend.edgecolor'] = 'black'

    #Titles
    rcParams['axes.titlesize'] = 24
   
    # Figure background
    rcParams['figure.facecolor'] = 'white'
    rcParams['axes.facecolor'] = 'white'
    rcParams['savefig.facecolor'] = 'white'
    rcParams['savefig.edgecolor'] = 'white'

theme_custom()

mpl.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 250,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.linewidth": 0.5,
    "font.size": 10
})

def mu_var_to_precision(mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    mu_c = np.clip(mu, 1e-4, 1.0 - 1e-4)
    var_c = np.clip(var, 1e-6, (mu_c * (1.0 - mu_c)) - 1e-7)
    phi = (mu_c * (1.0 - mu_c) / var_c) - 1.0
    return np.where(phi < 0, -1.0, phi)

def model_mean_curve(p: dict, x_interf: float, k_vec: np.ndarray, kmax: float) -> np.ndarray:
    floor = sigmoid(p["b0"] + p["b_i"] * x_interf)
    scale = np.exp(p["g0"] + p["g_i"] * x_interf)
    shape = np.exp(p.get("h0", 0.0))

    z = k_vec / kmax
    exponent = np.clip((scale * z) ** shape, 0.0, 60.0)
    sat = 1.0 - np.exp(-exponent)

    mu_n = floor + (1.0 - floor) * sat
    return np.clip(mu_n, 1e-4, 1 - 1e-4) * kmax

# ============================================================================
# Isolated Plotting Functions
# ============================================================================

def plot_standalone_weibull(p: dict, df: pd.DataFrame, truth_col: str, kmax: float, out_base: str):
    """
    Generates an isolated Weibull Saturation plot showing only a single median response curve.
    Forces figure aspect ratio to tightly match the physical scale of the axes.
    """
    gt = df[truth_col].values
    mu_raw = df["nn_mu"].values
    x = df["x"].values
    x_med = np.median(x)

    # Axis limits
    xmin, xmax = 0.0, kmax
    ymin, ymax = 0.0, kmax
    
    # Scale figure dimensions proportionally (1:1 if bounds are identical)
    x_range = xmax - xmin
    y_range = ymax - ymin
    base_size = 7.0
    figsize = (base_size, base_size * (y_range / x_range))

    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot components with explicit labels for the legend
    ax.scatter(gt, mu_raw, s=4, alpha=0.08, color="gray", rasterized=True, label="Empirical Observations")

    # Generate a smooth response trajectory at median interference
    k_smooth = np.linspace(xmin, xmax, 300)
    mu_smooth = model_mean_curve(p, x_med, k_smooth, kmax)
    
    ax.plot(k_smooth, mu_smooth, lw=2.5, color="#fdae61", label="Calibrated Response Curve")
    ax.plot([xmin, xmax], [ymin, ymax], "k--", lw=1, alpha=0.5, label="Identity Line")
    
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set(title="Weibull Saturation Curve", xlabel="True Count (k)", ylabel="NN Mean Output")
    
    # Standard legend placement
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none")

    fig.savefig(f"{out_base}_weibull_saturation.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved standalone Weibull curve: {out_base}_weibull_saturation.png")

def plot_density_ridgeplot(p: dict, df: pd.DataFrame, truth_col: str, kmax: float, out_base: str):
    """
    Generates a Beta density ridgeplot clipped between 0 and 8.
    Aligns the baselines cleanly with true integer values along the y-axis,
    forces a true square data aspect ratio, and removes the figure title.
    """
    gt = df[truth_col].values
    mu_raw = df["nn_mu"].values
    x = df["x"].values
    x_med = np.median(x)

    k_ints = np.round(gt).astype(int)
    
    MIN_K, MAX_K = 0, 8
    unique_k = [k for k in sorted(np.unique(k_ints)) if MIN_K <= k <= MAX_K and (k_ints == k).sum() > 5]

    median_phi_eff = {}
    for k_val in unique_k:
        mask = (k_ints == k_val)
        phi_base = np.exp(p["a0"] + p["a_f"] * k_val + p["a_i"] * x[mask])
        phi_nn_safe = np.maximum(1e-3, df["nn_phi"].values[mask])
        phi_eff = np.clip(phi_base * (phi_nn_safe ** p.get("alpha_v", 1.0)), 1, 1e4)
        median_phi_eff[k_val] = np.median(phi_eff)

    # Use a standard base size; ax.set_aspect handles the actual canvas geometry
    fig, ax = plt.subplots(figsize=(9, 9))
    
    y_grid_norm = np.linspace(0.001, 0.999, 400)
    y_grid_raw = y_grid_norm * kmax
    
    global_scale = 0.75 

    for k_val in unique_k:
        mask = (k_ints == k_val)
        empirical_mu_raw = mu_raw[mask]
        empirical_mu_clipped = np.clip(empirical_mu_raw, 0, MAX_K)

        counts, bins = np.histogram(empirical_mu_clipped, bins=40, range=(0, MAX_K), density=True)
        scale_factor = global_scale / (np.max(counts) + 1e-9)
        offset = float(k_val)

        # Alpha boosted to 0.55 for higher contrast, darker visual presence
        ax.bar(
            bins[:-1],
            counts * scale_factor,
            width=np.diff(bins),
            bottom=offset,
            align="edge",
            color="#4477AA",
            alpha=0.55,
            edgecolor="none"
        )

        mu_val_n = model_mean_curve(p, x_med, np.array([k_val]), kmax)[0] / kmax
        phi_val = median_phi_eff[k_val]
        pdf_vals = beta_dist.pdf(y_grid_norm, mu_val_n * phi_val, (1 - mu_val_n) * phi_val)

        pdf_scaled = (pdf_vals / kmax) * scale_factor
        valid_mask = (y_grid_raw <= MAX_K)
        ax.plot(y_grid_raw[valid_mask], pdf_scaled[valid_mask] + offset, color="#EE6677", lw=1.8)

    # Tighten data limits to prevent asymmetry from uneven padding strings
    ax.set_xlim(MIN_K - 0.2, MAX_K + 0.2)
    ax.set_ylim(MIN_K - 0.2, MAX_K + 1.0)
    
    # Hard anchor data space transformation to a 1:1 geometric ratio
    ax.set_aspect('equal', adjustable='box')
    
    ax.set_yticks(unique_k)
    ax.set_xticks(range(MIN_K, MAX_K + 1))
    
    # Title config removed for absolute print tracking lines
    ax.set(
        xlabel="NN Predicted Mean Output",
        ylabel="True Count Condition (k Baseline)"
    )
    
    fig.savefig(f"{out_base}_density_ridgeplot.png", bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved updated true geometric aspect ridgeplot: {out_base}_density_ridgeplot.png")

# ============================================================================
# Main Orchestration Loop
# ============================================================================

def main():
    params_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration_calibrated_v2.json")
    csv_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration.csv")
    out_base = params_path.replace(".json", "")

    # Load configuration parameters and dataset
    with open(params_path) as f:
        p = json.load(f)
    df = pd.read_csv(csv_path)
    
    kmax = float(p["K_MAX"])
    truth_col = TRUE_COUNT_COLUMN if TRUE_COUNT_COLUMN in df.columns else "gt_mu"

    # Pre-process dataframe structure
    df["mu_n"] = df["nn_mu"] / kmax
    df["var_n"] = df["nn_var"] / (kmax ** 2)
    df["nn_phi"] = mu_var_to_precision(df["mu_n"].values, df["var_n"].values)
    df["x"] = df["log_mean_rms_1000_1500"] - p["x_interf_center_mean"]

    # Generate standalone figures
    plot_standalone_weibull(p, df, truth_col, kmax, out_base)
    plot_density_ridgeplot(p, df, truth_col, kmax, out_base)

if __name__ == "__main__":
    main()