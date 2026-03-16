import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import frog_perch.config as config

# Import the single source of truth
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

# --- CONSTANTS & NUMERICAL GUARDS ---
EPS = 1e-6
K_MAX = 16.0  # The total number of slices/slots

def calibration_objective(params, y_mu_norm, nu_obs, q_k_matrix, x_interf):
    # Pack the scipy array into the dictionary format expected by sensor_model
    param_names = ['a0', 'a_f', 'a_i', 'ln_phi_nu', 'b0', 'b_i', 'g0', 'g_i', 'c0', 'c_u']
    p_dict = dict(zip(param_names, params))
    
    total_nll = 0.0
    
    # y_mu_norm is provided as [0,1], but sensor_model expects raw y [0,16]
    # So we multiply it back by K_MAX before passing it in.
    y_raw_array = y_mu_norm * K_MAX
    
    # We can vectorize the objective calculation slightly for speed
    for i in range(len(y_mu_norm)):
        log_liks = calculate_likelihood_vector(
            y_raw_array[i], 
            nu_obs[i], 
            x_interf[i], 
            p_dict, 
            k_max=K_MAX, 
            return_log_likelihoods=True # Critical flag!
        )
        # Multiply by the ground-truth weights and subtract from NLL
        total_nll -= np.sum(q_k_matrix[i, :] * log_liks)

    return total_nll

def main():
    # 1. Load Data
    csv_path = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration.csv")
    df = pd.read_csv(csv_path)

    # 2. Scale Covariates and Observed Intensities
    x_raw = df['log_mean_rms_1000_1500'].values
    x_mean_val = np.mean(x_raw)
    x_interf = x_raw - x_mean_val 
    
    y_mu_norm = np.clip(df['nn_mu'].values / K_MAX, 0.001, 0.999)
    
    norm_factor = (y_mu_norm * (1 - y_mu_norm))
    nu_obs = np.clip((df['nn_var'].values / (K_MAX**2)) / (norm_factor + EPS), 0.01, 0.99)
    
    q_k_matrix = np.array([eval(q) if isinstance(q, str) else q for q in df['q_k']])

    # 3. Optimized Initial Guesses
    init_guesses = [
        0.5, 0.1, 0.0, np.log(10.0), # alpha (Clarity)
        -3.5, 0.1,                   # beta (Noise Floor)
        3.0, 0.0,                    # gamma (Saturation Delta - Slower Rise)
        2.0, 0.1                     # C (Precision Modulation)
    ]

    print(f"Normalizing Intensity (K_MAX={K_MAX}) and centering noise at {x_mean_val:.2f}...")
    
    res = minimize(
        calibration_objective, 
        init_guesses, 
        args=(y_mu_norm, nu_obs, q_k_matrix, x_interf),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-5}
    )

    if res.success:
        print("\n=== CALIBRATION SUCCESSFUL ===")
        param_names = ['a0', 'a_f', 'a_i', 'ln_phi_nu', 'b0', 'b_i', 'g0', 'g_i', 'c0', 'c_u']
        fitted_params = dict(zip(param_names, res.x))
        fitted_params['x_interf_center_mean'] = x_mean_val
        fitted_params['K_MAX'] = K_MAX

        # Save results
        out_base = csv_path.replace(".csv", "_calibrated_v2")
        with open(f"{out_base}.json", 'w') as f:
            json.dump(fitted_params, f, indent=4)
            
        delta = np.exp(fitted_params['g0'])
        print(f"Fitted delta: {delta:.2f} (Counts per intensity slope)")
        print(f"Saved parameters to: {out_base}.json")
    else:
        print(f"Optimization failed: {res.message}")

if __name__ == "__main__":
    main()