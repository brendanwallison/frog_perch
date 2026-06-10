"""
calibrate.py

Optimizes the parameters of the Bayesian acoustic sensor model 
using extracted features from a validated model checkpoint run.
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Import the single source of truth
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

# --- NUMERICAL GUARDS ---
EPS = 1e-6

def calibration_objective(params, y_mu_norm, nu_obs, q_k_matrix, x_interf, k_max):
    """Calculates the negative log-likelihood across the feature tracking dataset."""
    # REVISED: Updated to match the 8-parameter specification from the formal latex document
    param_names = ['a0', 'a_f', 'a_i', 'alpha_v', 'b0', 'b_i', 'g0', 'g_i']
    p_dict = dict(zip(param_names, params))
    
    total_nll = 0.0
    
    # Scale normalized intensities [0, 1] back up to the current grid's maximum support
    y_raw_array = y_mu_norm * k_max
    
    for i in range(len(y_mu_norm)):
        log_liks = calculate_likelihood_vector(
            y_raw_array[i], 
            nu_obs[i], 
            x_interf[i], 
            p_dict, 
            k_max=k_max, 
            return_log_likelihoods=True
        )
        # Multiply by the ground-truth weights and subtract from NLL
        total_nll -= np.sum(q_k_matrix[i, :] * log_liks)

    return total_nll


def run_calibration(config_dict: dict, csv_path: str) -> dict | None:
    """
    Stateless external entry point to fit the sensor model parameters.
    Returns the fitted parameters dictionary or None if optimization fails.
    """
    if not os.path.exists(csv_path):
        print(f"[ERROR] Extracted calibration features CSV not found at: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    # Set maximum support boundary dynamically from the run configuration
    k_max = float(config_dict.get("MAX_BIN", 8))

    # 2. Scale Covariates and Observed Intensities
    x_raw = df['log_mean_rms_1000_1500'].values
    x_mean_val = np.mean(x_raw)
    x_interf = x_raw - x_mean_val 
    
    y_mu_norm = np.clip(df['nn_mu'].values / k_max, 0.001, 0.999)
    norm_factor = (y_mu_norm * (1 - y_mu_norm))
    nu_obs = np.clip((df['nn_var'].values / (k_max**2)) / (norm_factor + EPS), 0.01, 0.99)
    
    # Safely unpack list targets encoded as strings or native objects inside the CSV
    q_k_matrix = np.array([eval(q) if isinstance(q, str) else q for q in df['q_k']])

    # 3. REVISED: Initial guesses matching your 8 parameters 1-to-1
    # Layout order matches param_names layout in the objective function above
    init_guesses = [
        2.0,  0.0,  0.0,  1.0,  # a0, a_f, a_i, alpha_v (Trust parameter starts un-scaled at 1.0)
        -3.5, 0.1,             # b0, b_i (Noise floor parameters)
        1.0,  -0.1             # g0, g_i (Saturation scaling parameters)
    ]

    print(f"[INFO] Calibrating sensor model parameters (K_MAX={k_max})...")
    
    res = minimize(
        calibration_objective, 
        init_guesses, 
        args=(y_mu_norm, nu_obs, q_k_matrix, x_interf, k_max),
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-5}
    )

    if res.success:
        print("\n=== CALIBRATION SUCCESSFUL ===")
        param_names = ['a0', 'a_f', 'a_i', 'alpha_v', 'b0', 'b_i', 'g0', 'g_i']
        fitted_params = dict(zip(param_names, res.x))
        fitted_params['x_interf_center_mean'] = float(x_mean_val)
        fitted_params['K_MAX'] = k_max

        # Save calibration profile locally alongside the dataset checkpoint outputs
        out_json_path = csv_path.replace(".csv", "_calibrated_v2.json")
        with open(out_json_path, 'w') as f:
            json.dump(fitted_params, f, indent=4)
            
        print(f"Fitted trust parameter (alpha_v): {fitted_params['alpha_v']:.4f}")
        print(f"Saved parameters to: {out_json_path}")
        return fitted_params
    else:
        print(f"[ERROR] Optimization failed: {res.message}")
        return None