import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import os
import frog_perch.config as config

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import os
import frog_perch.config as config

def plot_multiband(csv_path):
    df = pd.read_csv(csv_path)
    sns.set_style("whitegrid")
    
    # Corrected terminology
    interference_band = "1000_1500"
    signal_band = "1500_2000"
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # VIEW A: The Masking Shadow (Interference Floor)
    # --------------------------------------------------
    ax = axes[0, 0]
    silence_df = df[df['gt_mu'] < 0.1].copy()
    
    sns.regplot(data=silence_df, x=f'log_mean_rms_{interference_band}', y='nn_mu', 
                color='red', scatter_kws={'alpha':0.1}, label='Interference Band Impact', ax=ax)
    sns.regplot(data=silence_df, x=f'log_mean_rms_{signal_band}', y='nn_mu', 
                color='blue', scatter_kws={'alpha':0.1}, label='Signal Band Self-Noise', ax=ax)
    
    ax.set_title("View A: False Positive Floor (Upward Masking Analysis)")
    ax.set_xlabel("Log Mean RMS")
    ax.set_ylabel("NN Intensity ($y_\mu$)")
    ax.legend()

    # VIEW B: Saturation under Interference
    # --------------------------------------------------
    ax = axes[0, 1]
    med_interf = df[f'log_mean_rms_{interference_band}'].median()
    
    bins_k = np.arange(17)
    for level, color in [('Low Interference', 'green'), ('High Interference', 'orange')]:
        mask = (df[f'log_mean_rms_{interference_band}'] <= med_interf) if 'Low' in level else (df[f'log_mean_rms_{interference_band}'] > med_interf)
        subset = df[mask]
        means, edges, _ = binned_statistic(subset['gt_mu'], subset['nn_mu'], bins=bins_k)
        ax.plot((edges[:-1]+edges[1:])/2, means, marker='o', label=level, color=color, linewidth=3)
    
    ax.set_title("View B: Saturation Curve Shifted by Interference")
    ax.set_xlabel("GT Count ($k$)")
    ax.set_ylabel("NN Predicted Intensity ($y_\mu$)")
    ax.legend()

    # VIEW C: Drone vs. Pulsed Performance
    # --------------------------------------------------
    ax = axes[1, 0]
    # We want to see if the "Constant Drone" (Interference) or "Pulsed Complexity" (Signal) drives error
    sns.scatterplot(data=df, x=f'var_rms_{interference_band}', y=np.abs(df['nn_mu']-df['gt_mu']), 
                    alpha=0.1, color='red', label='Interference Sparsity', ax=ax)
    sns.scatterplot(data=df, x=f'var_rms_{signal_band}', y=np.abs(df['nn_mu']-df['gt_mu']), 
                    alpha=0.1, color='blue', label='Signal Sparsity', ax=ax)
    
    ax.set_title("View C: Error vs. Temporal Sparsity")
    ax.set_xlabel("Variance of RMS (Sparsity)")
    ax.set_ylabel("Absolute Error")
    ax.legend()

    # VIEW D: Joint Error Density
    # --------------------------------------------------
    ax = axes[1, 1]
    h = ax.hist2d(df[f'log_mean_rms_{interference_band}'], df[f'log_mean_rms_{signal_band}'], 
                   weights=np.abs(df['nn_mu']-df['gt_mu']), bins=30, cmap='magma', norm=plt.cm.colors.LogNorm())
    plt.colorbar(h[3], ax=ax, label='Total Error Weight')
    ax.set_title("View D: Error Hotspots (Interference vs Signal Energy)")
    ax.set_xlabel("Log Interference Energy")
    ax.set_ylabel("Log Signal Energy")

    plt.tight_layout()
    save_path = csv_path.replace(".csv", "_insight_plots.png")
    plt.savefig(save_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    csv_file = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration.csv")
    plot_multiband(csv_file)