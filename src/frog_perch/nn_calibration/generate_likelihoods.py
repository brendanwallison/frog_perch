import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import frog_perch.config as config

# Import the single source of truth for our math
from frog_perch.nn_calibration.sensor_model import calculate_likelihood_vector

def calibrate_dataset(csv_path, params_path, output_path):
    print(f"Loading parameters from:\n  {params_path}")
    with open(params_path, 'r') as f:
        p = json.load(f)
        
    print(f"Loading uncalibrated data from:\n  {csv_path}")
    df = pd.read_csv(csv_path)
    
    k_max = p.get("K_MAX", 16.0)
    k_vec = np.arange(int(k_max) + 1)
    x_center = p.get("x_interf_center_mean", 0.0)

    print(f"Processing {len(df)} windows...")
    likelihoods = []
    
    for _, row in df.iterrows():
        # Center the noise covariate
        x_interf = row['log_mean_rms_1000_1500'] - x_center
        
        # Calculate clarity ratio (nu_obs) 
        y_mu_norm = np.clip(row['nn_mu'] / k_max, 0.001, 0.999)
        norm_factor = (y_mu_norm * (1 - y_mu_norm)) + 1e-6
        nu_obs = (row['nn_var'] / (k_max**2)) / norm_factor
        
        # Get the 17-bin probability vector using the shared sensor_model function
        lik_vec = calculate_likelihood_vector(row['nn_mu'], nu_obs, x_interf, p, k_max)
        likelihoods.append(lik_vec)

    # Save the probability vectors as new columns (p0, p1, ..., p16)
    lik_df = pd.DataFrame(likelihoods, columns=[f'p{k}' for k in k_vec])
    result_df = pd.concat([df.reset_index(drop=True), lik_df], axis=1)
    
    result_df.to_csv(output_path, index=False)
    print(f"Calibrated dataset saved to:\n  {output_path}")

if __name__ == "__main__":
    # Define default paths based on the established project structure
    default_input = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration.csv")
    default_params = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras_multiband_calibration_calibrated_v2.json")
    default_output = default_input.replace(".csv", "_likelihoods.csv")

    parser = argparse.ArgumentParser(description="Generate calibrated likelihood vectors for audio windows.")
    parser.add_argument("--input_csv", default=default_input, help="Raw extraction CSV (field data)")
    parser.add_argument("--params_json", default=default_params, help="Fitted calibration parameters")
    parser.add_argument("--output_csv", default=default_output, help="Where to save the dataset with p0-p16 columns")
    args = parser.parse_args()
    
    calibrate_dataset(args.input_csv, args.params_json, args.output_csv)