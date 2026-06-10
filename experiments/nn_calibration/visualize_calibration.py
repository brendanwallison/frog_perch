import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from scipy.special import expit as sigmoid
import frog_perch.config as config

# Import the single source of truth for the math
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

class CalibrationVisualizer:
    def __init__(self, params_path, csv_path):
        with open(params_path, 'r') as f:
            self.p = json.load(f)
        
        self.df = pd.read_csv(csv_path)
        self.k_vec = np.arange(17)
        self.k_max = self.p.get("K_MAX", 16.0)
        
        # Calculate noise statistics from the diagnostic CSV
        noise_raw = self.df['log_mean_rms_1000_1500'].values
        self.sigma = np.std(noise_raw)
        self.mu_noise = np.mean(noise_raw)

    def get_mu_y_norm(self, k, x_rel):
        """
        Retained locally ONLY for plotting the Generative Response curves in View 1.
        The actual likelihood math is handled by sensor_model.py.
        """
        noise_floor = sigmoid(self.p["b0"] + self.p["b_i"] * x_rel)
        delta = np.exp(self.p["g0"] - self.p["g_i"] * x_rel)
        return noise_floor + (1 - noise_floor) * np.tanh(k / delta)

    def generate_plots(self, output_path):
        fig = plt.figure(figsize=(22, 14))
        # Top row: Global Diagnostics | Bottom row: Comparative Scenarios
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
        plt.subplots_adjust(hspace=0.4, wspace=0.25)

        # --- VIEW 1: STATISTICAL RESPONSE CURVES ---
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.k_vec, self.k_vec, '--', color='gray', alpha=0.5, label='Perfect Parity')
        levels = [(-2*self.sigma, 'green', '-2σ (Quiet)'), 
                  (0, 'orange', 'μ (Avg)'), 
                  (2*self.sigma, 'red', '+2σ (Loud)')]
        
        for x_rel, col, lab in levels:
            mu_y_frogs = self.get_mu_y_norm(self.k_vec, x_rel) * self.k_max
            ax1.plot(self.k_vec, mu_y_frogs, marker='o', color=col, label=lab)
        
        ax1.set_title(f"A. Generative Response (σ={self.sigma:.2f})", loc='left', fontweight='bold')
        ax1.set_ylabel("Expected NN Predicted Count")
        ax1.legend(); ax1.grid(True, alpha=0.2)

        # --- VIEW 2: RESOLVING POWER HEATMAP ---
        ax2 = fig.add_subplot(gs[0, 1])
        y_grid = np.linspace(0.1, 15.9, 100)
        heatmap = np.zeros((len(y_grid), len(self.k_vec)))
        # Use baseline clarity for the heatmap
        nu_base = sigmoid(self.p["a0"])
        
        for i, y_val in enumerate(y_grid):
            # Using our centralized sensor_model math
            heatmap[i, :] = calculate_likelihood_vector(y_val, nu_base, 0, self.p, self.k_max)
            
        im = ax2.imshow(heatmap, aspect='auto', origin='lower', extent=[0, 16, 0, 16], cmap='magma')
        plt.colorbar(im, ax=ax2, label='Likelihood P(k|y)')
        ax2.set_title("B. Resolving Power @ μ Noise", loc='left', fontweight='bold')
        ax2.set_xlabel("Latent Count (k)"); ax2.set_ylabel("Observed NN Count")

        # --- VIEW 3: COMPARATIVE SCENARIO MATRIX ---
        # We compare Quiet (-2σ) vs Loud (+2σ) for three intensity levels
        scenario_gs = gs[1, :].subgridspec(1, 3)
        y_test_cases = [0.2, 2.5, 6.5]
        
        for i, y_test in enumerate(y_test_cases):
            ax_scen = fig.add_subplot(scenario_gs[i])
            
            # Plot Quiet vs Loud distributions using sensor_model.py
            lik_quiet = calculate_likelihood_vector(y_test, 0.2, -2*self.sigma, self.p, self.k_max)
            lik_loud = calculate_likelihood_vector(y_test, 0.2, 2*self.sigma, self.p, self.k_max)
            
            ax_scen.bar(self.k_vec - 0.2, lik_quiet, width=0.4, color='green', alpha=0.5, label='Quiet (-2σ)')
            ax_scen.bar(self.k_vec + 0.2, lik_loud, width=0.4, color='red', alpha=0.5, label='Loud (+2σ)')
            
            ax_scen.set_title(f"Observed y={y_test}", fontweight='bold')
            ax_scen.set_xlabel("k")
            ax_scen.set_ylim(0, 1.1)
            if i == 0: 
                ax_scen.set_ylabel("Likelihood P(k|data)")
                ax_scen.legend(fontsize=9)

        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        print(f"All diagnostic plots saved to: {output_path}")
        plt.show()

def main():
    p_json = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration_calibrated_v2.json")
    c_csv = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration.csv")
    
    viz = CalibrationVisualizer(p_json, c_csv)
    viz.generate_plots(p_json.replace(".json", "_comprehensive_diagnostics.png"))

if __name__ == "__main__":
    main()