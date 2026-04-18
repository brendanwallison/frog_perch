import os
import yaml
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Force non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from pathlib import Path
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (
    precision_recall_curve, f1_score, confusion_matrix, 
    average_precision_score, roc_auc_score
)

import configs.nn_config as config
from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.nn_training.dataset_builders import build_tf_val_dataset
from frog_perch.nn_models.model_utils import load_custom_model

# -----------------------------------------------------------------------------
# Loaders & Numerical Helpers
# -----------------------------------------------------------------------------

def load_normalization_stats(yaml_path: str) -> dict:
    """Safely load the generated normalization stats."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {yaml_path}. Run normalization first.")

def get_binned_stats(x, y, bins=20):
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    return bin_centers[valid], bin_means[valid]

# -----------------------------------------------------------------------------
# Spline / Trend Helpers
# -----------------------------------------------------------------------------

def add_raw_spline_trend(ax, x, y, label="LOWESS Fit", color='gold'):
    """
    Fits a LOWESS (Locally Weighted Scatterplot Smoothing) to raw data points.
    Efficiently captures the local trend of thousands of points without binning artifacts.
    """
    # Clean data (remove NaNs)
    mask = np.isfinite(x) & np.isfinite(y)
    x_clean, y_clean = x[mask], y[mask]
    
    if len(x_clean) < 10:
        return

    # To keep it efficient for very large datasets, we subsample for the fit 
    # but the line will remain smooth.
    if len(x_clean) > 5000:
        idx = np.random.choice(len(x_clean), 5000, replace=False)
        x_fit, y_fit = x_clean[idx], y_clean[idx]
    else:
        x_fit, y_fit = x_clean, y_clean

    # frac=0.2 defines the 'smoothness' (fraction of data used for each local fit)
    res = lowess(y_fit, x_fit, frac=0.2)
    
    # Sort by X for a continuous plot line
    ax.plot(res[:, 0], res[:, 1], color=color, linewidth=3, label=label, zorder=10)

# -----------------------------------------------------------------------------
# The Joint Variable & Error Flow Suite
# -----------------------------------------------------------------------------

def plot_joint_distribution_comparison(gt_mu, gt_var, pred_mu, pred_var, save_path=None):
    """Visualizes the 2D joint density of (Mean, Variance) for GT vs Pred."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    max_mu = max(np.max(gt_mu), np.max(pred_mu))
    
    # GT Density
    h1 = ax1.hist2d(gt_mu, gt_var, bins=50, norm=colors.LogNorm(), cmap='Blues')
    ax1.set_title(fr"Ground Truth Joint Density $(\mu_{{true}}, \sigma^2_{{true}})$")
    ax1.set_ylabel("Variance (Uncertainty)")
    fig.colorbar(h1[3], ax=ax1, label='Log10(Count)')

    # Pred Density
    h2 = ax2.hist2d(pred_mu, pred_var, bins=50, norm=colors.LogNorm(), cmap='Reds')
    ax2.set_title(fr"Predicted Joint Density $(\mu_{{pred}}, \sigma^2_{{pred}})$")
    fig.colorbar(h2[3], ax=ax2, label='Log10(Count)')

    # Theoretical Poisson-Binomial limit
    x_range = np.linspace(0, max_mu, 100)
    theoretical_max_var = x_range * (1 - x_range/16)
    for ax in [ax1, ax2]:
        ax.plot(x_range, theoretical_max_var, 'k--', alpha=0.5, label='Max PB Variance')
        ax.set_xlabel("Mean (Count)")
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_error_vector_flow(gt_mu, gt_var, pred_mu, pred_var, n_samples=500, save_path=None):
    """Shows displacement arrows from Truth to Prediction in Mean-Var space."""
    plt.figure(figsize=(12, 10))
    indices = np.random.choice(len(gt_mu), min(n_samples, len(gt_mu)), replace=False)
    
    u = pred_mu[indices] - gt_mu[indices]
    v = pred_var[indices] - gt_var[indices]
    
    plt.quiver(gt_mu[indices], gt_var[indices], u, v, angles='xy', scale_units='xy', 
               scale=1, color='gray', alpha=0.3, width=0.002, headwidth=3)
    
    plt.scatter(gt_mu[indices], gt_var[indices], c='blue', s=15, label='GT', alpha=0.6)
    plt.scatter(pred_mu[indices], pred_var[indices], c='red', s=15, label='Pred', alpha=0.6)
    
    # Global Centroids
    avg_gt = [np.mean(gt_mu), np.mean(gt_var)]
    avg_pred = [np.mean(pred_mu), np.mean(pred_var)]
    plt.annotate('', xy=avg_pred, xytext=avg_gt, arrowprops=dict(facecolor='black', width=2))
    plt.scatter(*avg_gt, c='blue', s=150, edgecolors='white', linewidth=2, zorder=5)
    plt.scatter(*avg_pred, c='red', s=150, edgecolors='white', linewidth=2, zorder=5)

    plt.title(fr"Error Flow: Displacement in $(\mu, \sigma^2)$ Space")
    plt.xlabel(r"Mean Count ($\mu$)")
    plt.ylabel(r"Variance ($\sigma^2$)")
    plt.grid(alpha=0.2)
    plt.legend()
    if save_path: plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_joint_error_density(gt_mu, pred_mu, gt_var, pred_var, save_path=None):
    """Heatmap of Mean Error vs Variance Error."""
    mu_error = pred_mu - gt_mu
    var_error = pred_var - gt_var
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(mu_error, var_error, bins=50, norm=colors.LogNorm(), cmap='magma')
    plt.colorbar(h[3], label='Log10(Windows)')
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)
    plt.title("Joint Error Density: Count Error vs. Uncertainty Error")
    plt.xlabel(r"Mean Error (Pred $\mu$ - GT $\mu$)")
    plt.ylabel(r"Variance Error (Pred $\sigma^2$ - GT $\sigma^2$)")
    if save_path: plt.savefig(save_path)
    plt.close()

# -----------------------------------------------------------------------------
# KDE & Calibration Plots
# -----------------------------------------------------------------------------

def plot_calibration_log_log(gt_mu, pred_mu, save_path=None):
    """Log1p Calibration Heatmap."""
    plt.figure(figsize=(10, 8))
    gt_log, pred_log = np.log1p(gt_mu), np.log1p(pred_mu)
    max_val = np.log1p(20) 
    bins = np.linspace(0, max_val, num=40)
    plt.hist2d(gt_log, pred_log, bins=[bins, bins], norm=colors.LogNorm(), cmap='viridis')
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.8, label='Ideal')
    plt.title(r"Log1p Calibration: $\ln(1 + \text{Pred})$ vs $\ln(1 + \text{GT})$")
    plt.xlabel("Log(1 + Ground Truth Count)"); plt.ylabel("Log(1 + Predicted Count)")
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_calibration_kde(gt_mu, pred_mu, bw_adjust=0.5, save_path=None):
    """Mean vs Mean Calibration KDE with Raw Fit Spline."""
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=gt_mu, y=pred_mu, fill=True, thresh=0.02, levels=50, cmap="mako", bw_adjust=bw_adjust)
    
    limit = min(10, max(np.max(gt_mu), np.max(pred_mu)))
    plt.plot([0, limit], [0, limit], 'r--', alpha=0.7, label='Perfect Calibration')
    
    # Use RAW spline fit
    add_raw_spline_trend(plt.gca(), gt_mu, pred_mu, label="Calibration Spline (LOWESS)")
    
    plt.title(f"Mean Calibration (KDE + Spline, bw={bw_adjust})")
    plt.xlabel("Ground Truth Expected Count"); plt.ylabel("Predicted Expected Count")
    plt.xlim(0, limit); plt.ylim(0, limit)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_variance_alignment_kde(gt_var, pred_var, bw_adjust=0.5, save_path=None):
    """Variance vs Variance Alignment KDE with Raw Fit Spline."""
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=gt_var, y=pred_var, fill=True, thresh=0.02, levels=50, cmap="rocket", bw_adjust=bw_adjust)
    
    limit = max(np.max(gt_var), np.max(pred_var))
    plt.plot([0, limit], [0, limit], 'r--', alpha=0.7, label='Perfect Uncertainty Alignment')
    
    # Use RAW spline fit
    add_raw_spline_trend(plt.gca(), gt_var, pred_var, label="Uncertainty Spline (LOWESS)", color='cyan')
    
    plt.title(f"Uncertainty Alignment (KDE + Spline, bw={bw_adjust})")
    plt.xlabel("Ground Truth Variance"); plt.ylabel("Predicted Variance")
    plt.xlim(0, limit); plt.ylim(0, limit)
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    if save_path: plt.savefig(save_path)
    plt.close()

def plot_variance_diagnostics(gt_mu, pred_mu, gt_var, pred_var, save_prefix=None):
    """Standard Reliability Plots with Splines."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    squared_error = (pred_mu - gt_mu)**2

    # Plot 1: Variance Alignment (Simple Heatmap version)
    ax1.hist2d(gt_var, pred_var, bins=40, norm=colors.LogNorm(), cmap='magma')
    limit = max(np.max(gt_var), np.max(pred_var))
    ax1.plot([0, limit], [0, limit], 'r--', label='Ideal (y=x)')
    add_raw_spline_trend(ax1, gt_var, pred_var, label="Variance Spline", color='gold')
    ax1.set_title("Label Var vs Pred Var"); ax1.set_xlabel("GT Variance"); ax1.set_ylabel("Predicted Variance")
    ax1.legend(); ax1.grid(alpha=0.2)

    # Plot 2: Reliability (Squared Error vs Pred Var)
    ax2.hist2d(pred_var, squared_error, bins=40, norm=colors.LogNorm(), cmap='viridis')
    limit_err = max(np.max(pred_var), np.max(squared_error))
    ax2.plot([0, limit_err], [0, limit_err], 'r--', label='Ideal: MSE = Var')
    add_raw_spline_trend(ax2, pred_var, squared_error, label="Reliability Spline", color='cyan')
    ax2.set_title("Reliability: Squared Error vs. Pred Var")
    ax2.set_xlabel("Predicted Variance"); ax2.set_ylabel("Actual Squared Error")
    ax2.legend(); ax2.grid(alpha=0.2)

    plt.tight_layout()
    if save_prefix: plt.savefig(f"{save_prefix}_variance_suite.png")
    plt.close()

# -----------------------------------------------------------------------------
# Metrics & Core Logic
# -----------------------------------------------------------------------------

def compute_and_save_metrics(y_true_slices, y_prob_slices, gt_count_dist, pred_count_dist, ckpt_name, split):
    bins = np.arange(17)
    gt_mu = np.sum(gt_count_dist * bins, axis=1)
    pred_mu = np.sum(pred_count_dist * bins, axis=1)
    gt_var = np.sum(gt_count_dist * (bins**2), axis=1) - (gt_mu**2)
    pred_var = np.sum(pred_count_dist * (bins**2), axis=1) - (pred_mu**2)
    
    gt_flat = (y_true_slices.flatten() > 0.5).astype(int)
    prob_flat = y_prob_slices.flatten()
    prec, rec, _ = precision_recall_curve(gt_flat, prob_flat)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-8)
    
    bin_means, _, _ = binned_statistic(gt_mu, pred_mu - gt_mu, statistic='mean', bins=10)
    
    metrics = {
        "slice_f1": np.max(f1s),
        "slice_auprc": average_precision_score(gt_flat, prob_flat),
        "slice_aucroc": roc_auc_score(gt_flat, prob_flat),
        "count_mae": np.mean(np.abs(gt_mu - pred_mu)),
        "count_bias": np.mean(pred_mu - gt_mu),
        "count_rmse": np.sqrt(np.mean((gt_mu - pred_mu)**2)),
        "count_ece": np.nanmean(np.abs(bin_means)),
        "variance_mae": np.mean(np.abs(gt_var - pred_var))
    }
    
    csv_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_{split}_metrics.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    print(f"\n[METRICS] Saved to {csv_path}\n{pd.Series(metrics)}")
    return metrics

def run_full_suite(ckpt_name, split='test'):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    model = load_custom_model(ckpt_path)
    
    config_dir = Path(config.__file__).parent
    norm_stats = load_normalization_stats(config_dir / "normalization.yaml")
    conf_params = {
        "duration_stats": norm_stats.get("duration", {}),
        "bandwidth_stats": norm_stats.get("bandwidth", {}),
        "logistic_params": getattr(config, "CONFIDENCE_LOGISTIC_PARAMS", {})
    }

    ds_obj = FrogPerchDataset(split_type=split, val_stride_sec=1.0, 
                              audio_dir=config.AUDIO_DIR, annotation_dir=config.ANNOTATION_DIR,
                              confidence_params=conf_params)
    tf_ds = build_tf_val_dataset(ds_obj, batch_size=config.BATCH_SIZE)
    
    all_pred_slice, all_pred_count, all_gt_slice, all_gt_count = [], [], [], []

    for x, y_dict in tf_ds:
        preds = model.predict(x, verbose=0)
        all_pred_slice.append(preds["slice_probs"])
        all_pred_count.append(preds["count_probs"])
        all_gt_slice.append(y_dict["slice"].numpy())
        all_gt_count.append(y_dict["count_probs"].numpy())

    y_prob = np.concatenate(all_pred_slice, axis=0)
    pred_count_dist = np.concatenate(all_pred_count, axis=0)
    y_true_slices = np.concatenate(all_gt_slice, axis=0)
    gt_count_dist = np.concatenate(all_gt_count, axis=0)

    compute_and_save_metrics(y_true_slices, y_prob, gt_count_dist, pred_count_dist, ckpt_name, split)

    bins = np.arange(17)
    gt_mu = np.sum(gt_count_dist * bins, axis=1)
    pred_mu = np.sum(pred_count_dist * bins, axis=1)
    gt_var = np.sum(gt_count_dist * (bins**2), axis=1) - (gt_mu**2)
    pred_var = np.sum(pred_count_dist * (bins**2), axis=1) - (pred_mu**2)

    prefix = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_{split}")
    
    # Restored & Updated Plots
    plot_joint_distribution_comparison(gt_mu, gt_var, pred_mu, pred_var, f"{prefix}_joint_dist.png")
    plot_error_vector_flow(gt_mu, gt_var, pred_mu, pred_var, n_samples=500, save_path=f"{prefix}_flow.png")
    plot_joint_error_density(gt_mu, pred_mu, gt_var, pred_var, f"{prefix}_error_density.png")
    plot_calibration_log_log(gt_mu, pred_mu, f"{prefix}_cal_log1p.png")
    plot_calibration_kde(gt_mu, pred_mu, bw_adjust=0.6, save_path=f"{prefix}_cal_kde.png")
    plot_variance_alignment_kde(gt_var, pred_var, bw_adjust=0.6, save_path=f"{prefix}_var_kde.png")
    plot_variance_diagnostics(gt_mu, pred_mu, gt_var, pred_var, save_prefix=prefix)

if __name__ == "__main__":
    run_full_suite("best.keras", split='test')