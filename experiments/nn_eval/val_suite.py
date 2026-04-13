import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from scipy.stats import binned_statistic
import matplotlib.colors as colors
import frog_perch.config as config
from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.training.dataset_builders import build_tf_val_dataset
from frog_perch.metrics.slice_to_count_metrics import targets_to_soft_counts, slices_to_soft_counts

def load_checkpointed_model(ckpt_path):
    # You must provide the custom objects so Keras can deserialize the model
    from frog_perch.metrics.slice_to_count_metrics import (
        SliceLossWithSoftCountKL, SliceToCountKLDivergence, 
        SliceExpectedCountMAE, SliceEMD, SliceExpectedRecall, 
        SliceExpectedPrecision, SliceExpectedBinaryAccuracy,
        SliceToCountWrapper
    )
    from frog_perch.metrics.count_metrics import (
        ExpectedRecall, ExpectedPrecision, ExpectedBinaryAccuracy, 
        EMD, ExpectedCountMAE
    )

    custom_objects = {
        "SliceToCountWrapper": SliceToCountWrapper,
        "SliceLossWithSoftCountKL": SliceLossWithSoftCountKL,
        "SliceToCountKLDivergence": SliceToCountKLDivergence,
        "SliceExpectedCountMAE": SliceExpectedCountMAE,
        "SliceEMD": SliceEMD,
        "SliceExpectedRecall": SliceExpectedRecall,
        "SliceExpectedPrecision": SliceExpectedPrecision,
        "SliceExpectedBinaryAccuracy": SliceExpectedBinaryAccuracy,
        "ExpectedCountMAE": ExpectedCountMAE,
        "EMD": EMD,
        "ExpectedRecall": ExpectedRecall,
        "ExpectedPrecision": ExpectedPrecision,
        "ExpectedBinaryAccuracy": ExpectedBinaryAccuracy,
    }
    
    return tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)

def plot_calibration_curve(gt_mu, pred_mu, save_path=None):
    """
    Visualizes the relationship between Ground Truth expected counts 
    and Model Predicted expected counts.
    """
    plt.figure(figsize=(10, 6))
    
    # 1. Scatter plot of all windows (with jitter to see density)
    jitter = np.random.normal(0, 0.05, size=len(gt_mu))
    plt.scatter(gt_mu + jitter, pred_mu, alpha=0.1, s=10, label='Windows', color='teal')
    
    # 2. Perfect calibration line
    max_val = max(np.max(gt_mu), np.max(pred_mu))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration', linewidth=2)
    
    # 3. Trend line (running average)
    from scipy.stats import binned_statistic
    bin_means, bin_edges, _ = binned_statistic(gt_mu, pred_mu, statistic='mean', bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    plt.plot(bin_centers, bin_means, 'gold', marker='o', label='Model Trend', linewidth=3)

    plt.title("Count Calibration: Predicted vs Ground Truth Expected Count")
    plt.xlabel("Ground Truth Expected Count (Sum of Weights)")
    plt.ylabel("Predicted Expected Count (Sum of Sigmoids)")
    plt.grid(alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Calibration plot saved to {save_path}")
    plt.show()

def plot_calibration_heatmap(gt_mu, pred_mu, save_path=None):
    """
    Creates a 2D histogram heatmap of Predicted vs GT counts.
    Uses a Logarithmic scale for color intensity to handle class imbalance.
    """
    plt.figure(figsize=(10, 8))
    
    # 1. 2D Histogram (Heatmap)
    # bins=40 provides a good balance of resolution and density
    h = plt.hist2d(gt_mu, pred_mu, bins=40, norm=colors.LogNorm(), cmap='viridis')
    
    # Add a colorbar to explain the density
    cbar = plt.colorbar(h[3])
    cbar.set_label('Number of Windows (Log Scale)', rotation=270, labelpad=15)

    # 2. Perfect calibration line (y = x)
    max_val = max(np.max(gt_mu), np.max(pred_mu))
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Calibration', linewidth=2)
    
    # 3. Trend line (running average)
    # This helps see the systemic bias regardless of individual point density
    bin_means, bin_edges, _ = binned_statistic(gt_mu, pred_mu, statistic='mean', bins=20)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Filter out bins with no data to avoid line breaks
    valid = ~np.isnan(bin_means)
    plt.plot(bin_centers[valid], bin_means[valid], 'gold', marker='o', 
             label='Model Trend', linewidth=3, markersize=8, markeredgecolor='black')

    plt.title("Count Calibration Heatmap: Predicted vs Ground Truth")
    plt.xlabel("Ground Truth Expected Count (Sum of Weights)")
    plt.ylabel("Predicted Expected Count (Sum of Sigmoids)")
    plt.grid(alpha=0.2)
    plt.legend(facecolor='white', framealpha=0.8)
    
    if save_path:
        plt.savefig(save_path)
    plt.close() # Clean up memory

def get_binned_stats(x, y, bins=20):
    """Computes mean and centers for binned data, filtering out empty bins."""
    bin_means, bin_edges, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    valid = ~np.isnan(bin_means)
    return bin_centers[valid], bin_means[valid]



def plot_joint_distribution_comparison(gt_mu, gt_var, pred_mu, pred_var, save_path=None):
    """
    Visualizes the 2D joint density of (Mean, Variance) for both GT and Pred.
    Shows how the 'Shape of Uncertainty' shifts between labels and model.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), sharex=True, sharey=True)
    
    # Define common bounds for the axes
    max_mu = max(np.max(gt_mu), np.max(pred_mu))
    max_var = max(np.max(gt_var), np.max(pred_var))
    
    # Plot 1: Ground Truth Joint Density
    h1 = ax1.hist2d(gt_mu, gt_var, bins=50, norm=colors.LogNorm(), cmap='Blues')
    ax1.set_title("Ground Truth Joint Density $(\mu_{true}, \sigma^2_{true})$")
    ax1.set_xlabel("Mean (Count)")
    ax1.set_ylabel("Variance (Uncertainty)")
    fig.colorbar(h1[3], ax=ax1, label='Log10(Count)')

    # Plot 2: Prediction Joint Density
    h2 = ax2.hist2d(pred_mu, pred_var, bins=50, norm=colors.LogNorm(), cmap='Reds')
    ax2.set_title("Predicted Joint Density $(\mu_{pred}, \sigma^2_{pred})$")
    ax2.set_xlabel("Mean (Count)")
    fig.colorbar(h2[3], ax=ax2, label='Log10(Count)')

    # Add theoretical Poisson-Binomial boundary 
    # (Max variance for a given mean is mu * (1 - mu/n))
    x_range = np.linspace(0, max_mu, 100)
    theoretical_max_var = x_range * (1 - x_range/16) # Assuming 16 slices
    for ax in [ax1, ax2]:
        ax.plot(x_range, theoretical_max_var, 'k--', alpha=0.5, label='Max PB Variance')
        ax.grid(alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_variance_diagnostics(gt_mu, pred_mu, gt_var, pred_var, save_prefix=None):
    """
    1. Variance Calibration: GT Var vs Pred Var.
    2. Model Reliability: Squared Error vs Pred Var (Ideal is y=x).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    
    # Pre-calculate errors
    squared_error = (pred_mu - gt_mu)**2

    # --- Plot 1: Variance Calibration (Internal Consistency) ---
    h1 = ax1.hist2d(gt_var, pred_var, bins=40, norm=colors.LogNorm(), cmap='magma')
    fig.colorbar(h1[3], ax=ax1, label='Log(Count)')
    
    # Theoretical Ideal
    limit = max(np.max(gt_var), np.max(pred_var))
    ax1.plot([0, limit], [0, limit], 'r--', label='Ideal (y=x)', alpha=0.8)
    
    # Actual Trend
    centers, means = get_binned_stats(gt_var, pred_var)
    ax1.plot(centers, means, color='gold', marker='o', linewidth=3, 
             label='Variance Trend', markeredgecolor='black')
    
    ax1.set_title("Uncertainty Alignment: Label Var vs Pred Var")
    ax1.set_xlabel("Ground Truth Variance (from Reconciled Slices)")
    ax1.set_ylabel("Predicted Variance (Model Entropy)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Plot 2: Reliability (The "Truth Serum") ---
    # We plot Squared Error vs. Predicted Variance
    h2 = ax2.hist2d(pred_var, squared_error, bins=40, norm=colors.LogNorm(), cmap='viridis')
    fig.colorbar(h2[3], ax=ax2, label='Log(Count)')
    
    # Theoretical Ideal: MSE should equal Predicted Variance
    limit_err = max(np.max(pred_var), np.max(squared_error))
    ax2.plot([0, limit_err], [0, limit_err], 'r--', label='Ideal: MSE = Var', alpha=0.8)
    
    # Actual Trend
    centers_err, means_mse = get_binned_stats(pred_var, squared_error)
    ax2.plot(centers_err, means_mse, color='cyan', marker='s', linewidth=3, 
             label='Actual MSE', markeredgecolor='black')
    
    ax2.set_title("Reliability: Squared Error vs. Predicted Variance")
    ax2.set_xlabel("Predicted Variance (Model's Expected Error)")
    ax2.set_ylabel("Actual Squared Error")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_prefix:
        path = f"{save_prefix}_variance_suite.png"
        plt.savefig(path)
        print(f"[DIAGNOSTIC] Saved variance plots to {path}")
    plt.show()

def plot_error_vector_flow(gt_mu, gt_var, pred_mu, pred_var, n_samples=500, save_path=None):
    """
    Visualizes the displacement from Truth (Blue) to Prediction (Red).
    Arrows show the 'Force' and 'Direction' of model error in (Mean, Var) space.
    """
    plt.figure(figsize=(12, 10))
    
    # 1. Subsample to prevent a 'mess' of arrows
    indices = np.arange(len(gt_mu))
    np.random.shuffle(indices)
    idx = indices[:n_samples]
    
    # 2. Plot the Vectors (Quiver)
    # U, V are the displacements in X and Y directions
    u = pred_mu[idx] - gt_mu[idx]
    v = pred_var[idx] - gt_var[idx]
    
    # quiver(X, Y, U, V)
    plt.quiver(gt_mu[idx], gt_var[idx], u, v, 
               angles='xy', scale_units='xy', scale=1, 
               color='gray', alpha=0.3, width=0.002, headwidth=3)
    
    # 3. Plot start and end points for context
    plt.scatter(gt_mu[idx], gt_var[idx], c='blue', s=15, label='Ground Truth', alpha=0.6)
    plt.scatter(pred_mu[idx], pred_var[idx], c='red', s=15, label='Model Prediction', alpha=0.6)
    
    # 4. Highlight the Global Drift (Mean Vector)
    avg_gt = [np.mean(gt_mu), np.mean(gt_var)]
    avg_pred = [np.mean(pred_mu), np.mean(pred_var)]
    
    plt.annotate('', xy=avg_pred, xytext=avg_gt,
                 arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=10))
    plt.scatter(*avg_gt, c='blue', s=150, edgecolors='white', linewidth=2, zorder=5)
    plt.scatter(*avg_pred, c='red', s=150, edgecolors='white', linewidth=2, zorder=5)
    plt.text(avg_gt[0], avg_gt[1]-0.1, "Global GT Centroid", ha='center', fontweight='bold')
    plt.text(avg_pred[0], avg_pred[1]+0.1, "Global Pred Centroid", ha='center', fontweight='bold')

    plt.title(f"Error Flow: Displacement in $(\mu, \sigma^2)$ Space (N={n_samples})")
    plt.xlabel("Mean Count ($\mu$)")
    plt.ylabel("Variance ($\sigma^2$)")
    plt.grid(alpha=0.2)
    plt.legend()
    
    # Poisson-Binomial Boundary
    x_range = np.linspace(0, max(np.max(gt_mu), np.max(pred_mu)), 100)
    plt.plot(x_range, x_range * (1 - x_range/16), 'k--', alpha=0.4, label='PB Limit')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"[DIAGNOSTIC] Vector flow plot saved to {save_path}")
    plt.show()

def run_full_suite(ckpt_name, split='val'):
    # 1. Load Model
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    model = load_checkpointed_model(ckpt_path)
    
    # 2. Setup Dataset
    dataset_kwargs = dict(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        random_seed=config.RANDOM_SEED,
        label_mode='slice',  # Hardcode or pull from config
        q2_confidence=getattr(config, "Q2_CONFIDENCE", 0.75),
        use_continuous_confidence=getattr(config, "USE_CONTINUOUS_CONFIDENCE", True),
        confidence_params=getattr(config, "CONFIDENCE_PARAMS", {})
    )

    # 2. Instantiate with the correct split and stride
    # For diagnostics, we want the deterministic fixed-stride index
    ds_obj = FrogPerchDataset(
        split_type=split, 
        val_stride_sec=getattr(config, "VAL_STRIDE_SEC", 1.0),
        **dataset_kwargs
    )
    tf_ds = build_tf_val_dataset(ds_obj, batch_size=config.BATCH_SIZE)
    
    all_logits = []
    all_y_true = [] # These are just the 16-slice labels from the dataset

    for x, y in tf_ds:
        all_logits.append(model.predict(x, verbose=0))
        all_y_true.append(y.numpy())

    y_logits = np.concatenate(all_logits, axis=0)      # Shape (N, 16)
    y_true_slices = np.concatenate(all_y_true, axis=0) # Shape (N, 16)

    # --- Part A: Slice Localization ---
    y_prob = tf.nn.sigmoid(y_logits).numpy()
    analyze_slices(y_true_slices, y_prob)

    # --- Part B: Window Count Calibration ---
    # Since the dataset only gives us slices, we generate the distributions here
    # Use the TF-based utilities we already have
    gt_count_dist = targets_to_soft_counts(y_true_slices, max_bin=16).numpy()
    pred_count_dist = slices_to_soft_counts(y_logits, max_bin=16).numpy()

    # Pass the reconstructed distributions to your analyzer
    analyze_counts(gt_count_dist, pred_count_dist)

    bins = np.arange(17)
    gt_mu = np.sum(gt_count_dist * bins, axis=1)
    pred_mu = np.sum(pred_count_dist * bins, axis=1)

    gt_var = np.sum(gt_count_dist * (bins**2), axis=1) - (gt_mu**2)
    pred_var = np.sum(pred_count_dist * (bins**2), axis=1) - (pred_mu**2)

    # 4. Call the Diagnostic Plots
    save_prefix = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_{split}")
    
    # Plot 1: The Mean Calibration (Heatmap)
    plot_variance_diagnostics(gt_mu, pred_mu, gt_var, pred_var, save_prefix=save_prefix)

    plot_joint_error_density(gt_mu, pred_mu, gt_var, pred_var, save_path=f"{save_prefix}__mu_v.png")
    
    plot_joint_distribution_comparison(gt_mu, gt_var, pred_mu, pred_var, save_path=f"{save_prefix}_joint_density.png")

    plot_error_vector_flow(gt_mu, gt_var, pred_mu, pred_var, n_samples=500, save_path=f"{save_prefix}_joint_density_error_flow.png")

    plot_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_calibration_heatmap.png")
    plot_calibration_heatmap(gt_mu, pred_mu, save_path=plot_path)
    plot_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_calibration_jitter.png")
    plot_calibration_curve(gt_mu, pred_mu, save_path=plot_path)


def plot_joint_error_density(gt_mu, pred_mu, gt_var, pred_var, save_path=None):
    """
    Plots the joint density of Mean Error and Variance Error.
    Identifies if systemic biases in count are linked to systemic biases in uncertainty.
    """
    # Calculate Residuals (Errors)
    mu_error = pred_mu - gt_mu
    var_error = pred_var - gt_var

    plt.figure(figsize=(10, 8))
    
    # Joint Density Heatmap
    # We center the bins around 0 to see the 'Cross of Truth'
    limit_mu = np.max(np.abs(mu_error))
    limit_var = np.max(np.abs(var_error))
    
    h = plt.hist2d(mu_error, var_error, bins=50, 
                   norm=colors.LogNorm(), cmap='magma')
    
    plt.colorbar(h[3], label='Log10(Number of Windows)')

    # Reference lines (The Zero-Error Cross)
    plt.axhline(0, color='white', linestyle='--', alpha=0.5)
    plt.axvline(0, color='white', linestyle='--', alpha=0.5)

    # Annotate Quadrants
    plt.text(limit_mu*0.7, limit_var*0.7, "Overcount / Underconfident", color='white', fontsize=8, ha='center')
    plt.text(-limit_mu*0.7, limit_var*0.7, "Undercount / Underconfident", color='white', fontsize=8, ha='center')
    plt.text(-limit_mu*0.7, -limit_var*0.7, "Undercount / Overconfident", color='white', fontsize=8, ha='center')
    plt.text(limit_mu*0.7, -limit_var*0.7, "Overcount / Overconfident", color='white', fontsize=8, ha='center')

    plt.title("Joint Error Density: Count Error vs. Uncertainty Error")
    plt.xlabel("Mean Error (Pred $\mu$ - GT $\mu$)")
    plt.ylabel("Variance Error (Pred $\sigma^2$ - GT $\sigma^2$)")
    plt.grid(alpha=0.2)

    if save_path:
        plt.savefig(save_path)
    plt.show()

def analyze_slices(gt, prob):
    # Flatten to treat every 312ms slice as a binary classification task
    gt_flat = (gt.flatten() > 0.5).astype(int)
    prob_flat = prob.flatten()
    
    precision, recall, thresholds = precision_recall_curve(gt_flat, prob_flat)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    
    print(f"[SLICE] Best F1 Score: {f1_scores[best_idx]:.4f} at Threshold: {best_thresh:.3f}")
    
    # Confusion Matrix at best threshold
    preds_flat = (prob_flat > best_thresh).astype(int)
    cm = confusion_matrix(gt_flat, preds_flat)
    print(f"[SLICE] Confusion Matrix:\n{cm}")

def analyze_counts(gt_dist, pred_probs):
    # Ground Truth Expectation from the 17-bin PB distribution
    gt_mu = np.sum(gt_dist * np.arange(17), axis=1)
    
    # Predicted Expectation from the sum of slice probabilities
    pred_mu = np.sum(pred_probs, axis=1)
    
    mae = np.mean(np.abs(gt_mu - pred_mu))
    bias = np.mean(pred_mu - gt_mu)
    
    print(f"[COUNT] Window Expectation MAE: {mae:.4f}")
    print(f"[COUNT] Window Count Bias: {bias:.4f} (positive = overcounting)")

    # Variance Check: sum(p(1-p))
    pred_var = np.sum(pred_probs * (1 - pred_probs), axis=1)
    # Note: We can compare this to the actual variance of the GT distribution
    gt_var = np.sum(gt_dist * (np.arange(17)**2), axis=1) - (gt_mu**2)
    var_error = np.mean(np.abs(gt_var - pred_var))
    print(f"[COUNT] Variance Prediction Error: {var_error:.4f}")

if __name__ == "__main__":
    # Change this to your actual checkpoint filename
    target_ckpt = "pool=slice_loss=slice_x0=-3.0_k=1.0.keras" 
    run_full_suite(target_ckpt, split='test')