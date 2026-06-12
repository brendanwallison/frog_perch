#!/usr/bin/env python3
"""
Agnostic MAP Optimization, Automated Diagnostics, and Non-Overwriting Run Manager.
"""
import argparse
import json
import re
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import arviz as az

import numpyro
from numpyro.infer import SVI, Trace_ELBO, Predictive
from numpyro.infer.autoguide import AutoDelta
import numpyro.optim as optim

from frog_perch.stat_models.data.data_loading import load_detector_csvs
from frog_perch.stat_models.data.data_prep_for_rainfall_model import prepare_numpyro_data_hydrological
from frog_perch.stat_models.numpyro.call_intensity_profile_rain_hill import model

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_next_run_dir(base_output_dir):
    """
    Scans the base output folder and automatically creates an incremented 
    subfolder (e.g., run_001, run_002) to prevent overwriting.
    """
    base_path = Path(base_output_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    existing_runs = [d.name for d in base_path.iterdir() if d.is_dir() and re.match(r"^run_\d{3}$", d.name)]
    if not existing_runs:
        next_run_id = 1
    else:
        run_numbers = [int(name.split("_")[1]) for name in existing_runs]
        next_run_id = max(run_numbers) + 1
        
    run_dir = base_path / f"run_{next_run_id:03d}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def calculate_pearson_acf(w_obs, lambd, phi_nb, max_lag=14):
    """Calculates the temporal autocorrelation of Pearson residuals."""
    # Expected variance under Negative Binomial 2 model
    expected_var = lambd + (lambd**2) / phi_nb
    
    # Compress the matrix observation history to a 1D sequence across windows
    # to measure temporal residual drift
    y_true = w_obs.mean(axis=1) 
    y_pred = lambd
    
    # Compute Pearson Residuals
    residuals = (y_true - y_pred) / np.sqrt(expected_var + 1e-10)
    
    # Calculate ACF across specified lags
    acf_vals = []
    mean_res = np.mean(residuals)
    var_res = np.var(residuals) + 1e-10
    norm_res = residuals - mean_res
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf_vals.append(1.0)
        else:
            covariance = np.mean(norm_res[:-lag] * norm_res[lag:])
            acf_vals.append(covariance / var_res)
            
    return np.array(acf_vals)

def run_map_optimization(model_fn, data_dict, num_steps=5000, seed=0):
    guide = AutoDelta(model_fn)
    optimizer = optim.Adam(step_size=0.01)
    svi = SVI(model_fn, guide, optimizer, loss=Trace_ELBO())
    
    rng_key = jax.random.PRNGKey(seed)
    svi_result = svi.run(rng_key, num_steps, data_dict)
    
    return guide, svi_result.params, svi_result.losses

def main():
    parser = argparse.ArgumentParser(description="Agnostic JAX MAP Pipeline with Diagnostics")
    default_config = Path("configs/call_intensity_splines_rainfall.yaml")
    parser.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Handle non-overwriting nested subfolder scheme
    run_dir = get_next_run_dir(cfg["output_dir"])
    print(f"📁 Target execution directory initialized: {run_dir}")
    
    # Document current run config state inside the run directory for provenance
    with open(run_dir / "run_config.yaml", "w") as f:
        yaml.dump(cfg, f)
    
    # --- 1. Load Data ---
    df_raw = load_detector_csvs(Path(cfg["csv_dir"]))
    df_rain = pd.read_csv(Path(cfg["rain_csv"]))
    with open(Path(cfg["calibration_json_path"]), 'r') as f:
        calib_params = json.load(f)
        
    numpyro_data, windows_df, params = prepare_numpyro_data_hydrological(
        df_raw, df_rain, calibration_params=calib_params,
        knot_spacing_diel_min=cfg["knot_spacing_diel_min"],
        burn_in_days=cfg.get("burn_in_days", 14),
        w_fraction=cfg.get("w_fraction", 0.0167),
    )
    
    data_dict = {
        "w_obs":         jnp.array(numpyro_data["w_obs"]),
        "precip_daily":  jnp.array(numpyro_data["precip_daily"]),
        "B_diel":        jnp.array(numpyro_data["B_diel"]),
        "N":             numpyro_data["N"],
        "day_idx":       jnp.array(numpyro_data["day_idx"]),
        "num_days":      numpyro_data["num_days"]
    }

    # --- 2. Run MAP Optimization ---
    guide, map_params, losses = run_map_optimization(model, data_dict, num_steps=5000, seed=cfg["seed"])
    
    # --- 3. Run Forward Predictive Pass ---
    predictive = Predictive(model, guide=guide, params=map_params, num_samples=1)
    predictions = predictive(jax.random.PRNGKey(0), data_dict)
    
    # --- 4. Core Diagnostic Calculations ---
    print("📈 Extracting and evaluating model fit diagnostics...")
    
    # A. Number of free parameters optimized (excluding fixed constants)
    num_params_k = len([k for k in map_params.keys() if not k.endswith("_raw_auto_scale")])
    num_data_points_n = data_dict["w_obs"].shape[0]
    
    # B. Pull calculated maximum likelihood state from predictive outputs
    # Negative unnormalized log-posterior at the mode
    final_neg_log_post = float(losses[-1]) 
    
    # C. Random Effects Variance Evaluation
    # Pull out whichever random effects array the model populated
    alpha_key = "alpha_day_raw_auto_loc" if "alpha_day_raw_auto_loc" in map_params else "alpha_day_auto_loc"
    if alpha_key in map_params:
        alpha_vals = np.array(map_params[alpha_key])
        alpha_variance = float(np.var(alpha_vals))
    else:
        alpha_variance = 0.0
        
    # D. Residual Autocorrelation Extraction
    # Reconstruct lambda rate and dispersion parameter from point mass optimization
    phi_nb_val = float(map_params["phi_auto_loc"]) if "phi_auto_loc" in map_params else 1.0
    
    # Pull tracked vector trajectories
    exported_tracks = {}
    for tracking_name, tracking_vector in predictions.items():
        if len(tracking_vector.shape) > 0:
            exported_tracks[tracking_name] = np.array(tracking_vector[0])
            
    # Calculate lambda sequence array directly
    log_lambda = (
        map_params["beta_0_auto_loc"] + 
        exported_tracks.get("total_influence_track", 0.0) + 
        exported_tracks.get("random_effects_track", 0.0)[data_dict["day_idx"]]
    )
    lambd_seq = np.array(jnp.exp(log_lambda))
    
    acf_residuals = calculate_pearson_acf(np.array(data_dict["w_obs"]), lambd_seq, phi_nb_val, max_lag=14)
    
    # E. Construct Classic BIC
    # BIC = k * ln(n) - 2 * ln(L)
    # Using unnormalized negative log posterior as proxy for optimized objective density
    bic_value = num_params_k * np.log(num_data_points_n) + 2 * final_neg_log_post

    # Save diagnostic suite to disk
    diagnostics_payload = {
        "final_loss": final_neg_log_post,
        "bic_score": bic_value,
        "alpha_day_variance": alpha_variance,
        "num_parameters_k": num_params_k,
        "num_observations_n": num_data_points_n,
        "residual_acf_lags_0_to_14": acf_residuals.tolist()
    }
    
    with open(run_dir / "fit_diagnostics.json", "w") as f:
        json.dump(diagnostics_payload, f, indent=4)

    # --- 5. Export Vectors & Coordinate Tracks ---
    np.savez(
        run_dir / "map_predictions.npz",
        losses=np.array(losses),
        precip_daily=np.array(data_dict["precip_daily"]),
        residual_acf=acf_residuals,
        **exported_tracks
    )
    
    df_raw.to_csv(run_dir / "merged_detector_data.csv", index=False)
    windows_df.to_csv(run_dir / "windowed_detector_data.csv", index=False)
    
    # Screen Summary Printout
    print(f"\n=== Run Diagnostics Summary ({run_dir.name}) ===")
    print(f"  Final Negative Log-Posterior Loss : {final_neg_log_post:.2f}")
    print(f"  Calculated BIC Score             : {bic_value:.2f}")
    print(f"  Random Effect α Variance         : {alpha_variance:.4f}")
    print(f"  Residual Lag-1 Autocorrelation    : {acf_residuals[1]:.4f}")
    print(f"💾 All outputs frozen successfully inside {run_dir}")

if __name__ == "__main__":
    main()