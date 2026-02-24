import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure src is in path if running from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

import frog_perch.config as config
from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.training.dataset_builders import build_tf_val_dataset

# Import custom metrics/losses for model loading
from frog_perch.metrics.slice_to_count_metrics import (
    SliceLossWithSoftCountKL, 
    SliceToCountKLDivergence,
    SliceExpectedCountMAE, 
    SliceEMD, 
    SliceExpectedRecall,
    SliceExpectedPrecision, 
    SliceExpectedBinaryAccuracy,
    SliceToCountWrapper 
)

# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def get_target_moments_numpy(binary_slices):
    """
    Calculates the Mean and Variance of the Ground Truth distribution.
    If labels are binary (0 or 1), Variance is 0.
    If labels are soft (0.5), Variance > 0.
    """
    # Mean = Sum of probabilities
    mu_target = np.sum(binary_slices, axis=1)
    # Variance = Sum of p * (1-p)
    var_target = np.sum(binary_slices * (1 - binary_slices), axis=1)
    sigma_target = np.sqrt(var_target)
    return mu_target, sigma_target

def get_pb_moments(slice_probs):
    """
    Computes analytical Mean and Variance for Poisson-Binomial.
    slice_probs: (N, T) array of probabilities.
    """
    mu = np.sum(slice_probs, axis=1)
    var = np.sum(slice_probs * (1 - slice_probs), axis=1)
    sigma = np.sqrt(var)
    return mu, sigma

def get_pb_pmf(slice_probs, max_bin=16):
    N, T = slice_probs.shape
    dp = np.zeros((N, max_bin + 1))
    dp[:, 0] = 1.0
    
    for t in range(T):
        p = slice_probs[:, t:t+1]
        dp_shift = np.hstack([np.zeros((N, 1)), dp[:, :-1]])
        dp = dp * (1 - p) + dp_shift * p
        
    return dp

# ---------------------------------------------------------
# Visualization Functions (Standardized: X=Pred, Y=Truth)
# ---------------------------------------------------------

def plot_mean_calibration(target_mu, pred_mu, save_path=None):
    """
    Mean Calibration (Bias).
    X-Axis: Predicted Mean
    Y-Axis: Observed (Target) Mean
    """
    # Sort by Prediction (X-axis)
    sort_idx = np.argsort(pred_mu)
    sorted_pred = pred_mu[sort_idx]
    sorted_true = target_mu[sort_idx]
    
    n_bins = 10
    if len(sorted_pred) < n_bins * 2: return
    bins = np.array_split(np.arange(len(sorted_pred)), n_bins)
    
    bin_pred = [np.mean(sorted_pred[b]) for b in bins]
    bin_true = [np.mean(sorted_true[b]) for b in bins]
        
    plt.figure(figsize=(6, 6))
    max_val = max(max(bin_pred), max(bin_true)) + 0.5
    plt.plot([0, max_val], [0, max_val], 'k--', label='Ideal')
    plt.plot(bin_pred, bin_true, 'o-', lw=2, color='green', label='Model')
    
    plt.xlabel("Predicted Mean Count")
    plt.ylabel("Observed Mean Count")
    plt.title("Mean Calibration (Bias)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_variance_matching(target_sigma, pred_sigma, save_path=None):
    """
    Variance Matching.
    X-Axis: Predicted Sigma
    Y-Axis: Target Sigma
    """
    # Sort by Prediction (X-axis)
    sort_idx = np.argsort(pred_sigma)
    sorted_pred = pred_sigma[sort_idx]
    sorted_target = target_sigma[sort_idx]
    
    n_bins = 10
    if len(sorted_pred) < n_bins * 2: return
    bins = np.array_split(np.arange(len(sorted_pred)), n_bins)
    
    bin_pred_sigma = [np.mean(sorted_pred[b]) for b in bins]
    bin_target_sigma = [np.mean(sorted_target[b]) for b in bins]
        
    plt.figure(figsize=(6, 6))
    max_val = max(max(bin_target_sigma), max(bin_pred_sigma)) + 0.1
    plt.plot([0, max_val], [0, max_val], 'k--', label='Ideal')
    plt.plot(bin_pred_sigma, bin_target_sigma, 'o-', lw=2, color='purple', label='Model')
    
    plt.xlabel("Predicted Sigma (Model Uncertainty)")
    plt.ylabel("Target Sigma (Label Noise)")
    plt.title("Variance Matching\n(Does model mimic label fuzziness?)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_spread_skill(target_mu, pred_mu, pred_sigma, save_path=None):
    """
    Spread-Skill.
    X-Axis: Predicted Sigma
    Y-Axis: Observed RMSE (Skill)
    """
    errors = np.abs(pred_mu - target_mu)
    
    # Sort by Prediction (X-axis)
    sort_idx = np.argsort(pred_sigma)
    sorted_sigma = pred_sigma[sort_idx]
    sorted_errors = errors[sort_idx]
    
    n_bins = 10
    if len(sorted_sigma) < n_bins * 2: return
    bins = np.array_split(np.arange(len(sorted_sigma)), n_bins)
    
    bin_mean_sigma = [np.mean(sorted_sigma[b]) for b in bins]
    bin_rmse_error = [np.sqrt(np.mean(sorted_errors[b]**2)) for b in bins]
        
    plt.figure(figsize=(6, 6))
    max_val = max(max(bin_mean_sigma), max(bin_rmse_error)) + 0.1
    plt.plot([0, max_val], [0, max_val], 'k--', label='Ideal')
    plt.plot(bin_mean_sigma, bin_rmse_error, 'o-', lw=2, color='blue', label='Model')
    
    plt.xlabel("Predicted Sigma (Spread)")
    plt.ylabel("Observed Error RMSE (Skill)")
    plt.title("Spread-Skill Relationship\n(Is uncertainty honest about error?)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_pit(target_mu, pred_pmfs, save_path=None):
    pit_values = []
    n_samples = len(target_mu)
    
    for i in range(n_samples):
        # Round target mean to nearest int for PIT evaluation
        k = int(round(target_mu[i]))
        pmf = pred_pmfs[i]
        
        cdf_k = np.sum(pmf[:k+1])
        cdf_k_minus_1 = np.sum(pmf[:k]) if k > 0 else 0.0
        
        u = np.random.uniform(cdf_k_minus_1, cdf_k)
        pit_values.append(u)
        
    plt.figure(figsize=(8, 5))
    plt.hist(pit_values, bins=20, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axhline(1.0, color='red', linestyle='--', lw=2, label='Perfect')
    
    plt.xlabel("PIT Value")
    plt.ylabel("Density")
    plt.title("Randomized PIT Histogram")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# ---------------------------------------------------------
# Main Analysis
# ---------------------------------------------------------

def analyze(model_path):
    print(f"--- Loading Model: {model_path} ---")
    
    custom_objects = {
        'SliceLossWithSoftCountKL': SliceLossWithSoftCountKL,
        'SliceToCountKLDivergence': SliceToCountKLDivergence,
        'SliceExpectedCountMAE': SliceExpectedCountMAE,
        'SliceEMD': SliceEMD,
        'SliceExpectedRecall': SliceExpectedRecall,
        'SliceExpectedPrecision': SliceExpectedPrecision,
        'SliceExpectedBinaryAccuracy': SliceExpectedBinaryAccuracy,
        'SliceToCountWrapper': SliceToCountWrapper,
    }
    
    try:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"--- Reconstructing Validation Set (Seed {config.RANDOM_SEED}) ---")
    
    val_ds_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False, 
        pos_ratio=None,
        random_seed=config.RANDOM_SEED,
        label_mode='slice',
        val_stride_sec=getattr(config, "VAL_STRIDE_SEC", 1.0),
        q2_confidence=getattr(config, "Q2_CONFIDENCE", 0.75),
        equalize_q2_val=getattr(config, "EQUALIZE_Q2_VAL", False),
        use_continuous_confidence=getattr(config, "USE_CONTINUOUS_CONFIDENCE", False),
        confidence_params=getattr(config, "CONFIDENCE_PARAMS", {})
    )

    val_ds = build_tf_val_dataset(val_ds_obj, batch_size=config.BATCH_SIZE)
    
    print("--- Running Inference ---")
    all_logits = []
    all_target_mu = []
    all_target_sigma = []
    
    for batch_x, batch_y in tqdm(val_ds):
        logits = model.predict_on_batch(batch_x)
        all_logits.append(logits)
        
        # Calculate Target Moments directly from the batch labels
        mu_t, sigma_t = get_target_moments_numpy(batch_y.numpy())
        all_target_mu.append(mu_t)
        all_target_sigma.append(sigma_t)

    logits_flat = np.vstack(all_logits)       
    target_mu_flat = np.hstack(all_target_mu)
    target_sigma_flat = np.hstack(all_target_sigma)
    
    # Calculate Predicted Moments
    probs_flat = sigmoid(logits_flat)
    pred_mu, pred_sigma = get_pb_moments(probs_flat)
    pred_pmfs = get_pb_pmf(probs_flat, max_bin=16)

    # Metrics
    mae = np.mean(np.abs(pred_mu - target_mu_flat))
    rmse = np.sqrt(np.mean((pred_mu - target_mu_flat)**2))
    bias = np.mean(pred_mu - target_mu_flat)
    
    # Spread-Skill
    avg_pred_var = np.mean(pred_sigma**2)
    rmsv = np.sqrt(avg_pred_var)
    ssr = rmsv / rmse
    
    print("\n" + "="*40)
    print("        VALIDATION RESULTS      ")
    print("="*40)
    print(f"Bias:                  {bias:.4f}")
    print(f"MAE:                   {mae:.4f}")
    print(f"Spread-Skill Ratio:    {ssr:.4f}")
    print("="*40)

    # Plots
    base_name = os.path.splitext(model_path)[0]
    
    plot_mean_calibration(target_mu_flat, pred_mu, f"{base_name}_bias.png")
    plot_variance_matching(target_sigma_flat, pred_sigma, f"{base_name}_var_match.png")
    plot_spread_skill(target_mu_flat, pred_mu, pred_sigma, f"{base_name}_spread_skill.png")
    plot_pit(target_mu_flat, pred_pmfs, f"{base_name}_pit.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    args = parser.parse_args()
    if os.path.exists(args.model_path):
        analyze(args.model_path)
    else:
        print("Model not found")