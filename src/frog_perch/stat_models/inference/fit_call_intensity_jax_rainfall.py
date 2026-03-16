"""
JAX/NumPyro pipeline for call-intensity hydrological decay model.
"""
import argparse
import shutil  # NEW: for copying config
from pathlib import Path
import yaml
import json
import numpy as np
import arviz as az
import pandas as pd

# 1. Load Utilities
from frog_perch.stat_models.data.data_loading import load_detector_csvs
from frog_perch.stat_models.data.data_prep_for_rainfall_model import prepare_stan_data_hydrological
from frog_perch.stat_models.numpyro.call_intensity_profile_rain_hill import compile_and_run

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run JAX Call-Intensity Rain Model")
    
    # Define your default config path here
    default_config = Path("/home/breallis/dev/frog_perch/src/frog_perch/stat_models/inference/call_intensity_splines_rainfall.yaml") # Update this to your actual default path!
    
    parser.add_argument(
        "--config", 
        type=Path, 
        default=default_config, 
        help=f"Path to YAML config (default: {default_config})"
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # --- Path Resolution from Config ---
    csv_dir = Path(cfg["csv_dir"])
    rain_path = Path(cfg["rain_csv"])
    output_dir = Path(cfg["output_dir"])
    calibration_json_path = Path(cfg["calibration_json_path"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # NEW: Copy config file to output directory for reproducibility
    shutil.copy2(args.config, output_dir / "run_config.yaml")
    
    # --- 1. Load & Prepare Data ---
    print(f"🚀 Loading detector data from {csv_dir}...")
    df_raw = load_detector_csvs(csv_dir)
    
    print(f"🌧️ Loading rainfall data from {rain_path}...")
    df_rain = pd.read_csv(rain_path)
    
    print(f"⚙️ Loading calibration parameters from {calibration_json_path}...")
    with open(calibration_json_path, 'r') as f:
        calib_params = json.load(f)
    
    print("🛠️ Preparing Hydrological Decay data...")
    stan_data, windows_df, params = prepare_stan_data_hydrological(
        df_raw,
        df_rain,
        calibration_params=calib_params,
        bin_duration_sec=cfg["bin_duration_sec"],
        window_length_sec=cfg["window_length_sec"],
        knot_spacing_diel_min=cfg["knot_spacing_diel_min"],
        burn_in_days=cfg.get("burn_in_days", 14)
    )
    
    # --- 2. Run Inference (JAX) ---
    print("🔥 Starting Sampling (NumPyro - Rain Model)...")
    mcmc = compile_and_run(
        stan_data,
        num_warmup=cfg["warmup"],
        num_samples=cfg["sampling"],
        num_chains=cfg["chains"],
        seed=cfg["seed"]
    )
    
    # --- 3. Save Results ---
    print(f"💾 Saving results to {output_dir}...")
    
    # Convert to ArviZ InferenceData
    # UPDATED: Mapping the dual-scale variables and wetness states
    idata = az.from_numpyro(
        mcmc,
        constant_data={
            "w_obs": stan_data["w_obs"],
            "precip_daily": stan_data["precip_daily"]
        },
        dims={
            "trend_diel": ["time"],
            "lambda": ["time"],
            "alpha_day": ["day_id"],
            "rain_effect_fast": ["time"],
            "rain_effect_slow": ["time"],
            "total_rain_effect": ["time"],
            "daily_wetness_fast": ["day_id"],
            "daily_wetness_slow": ["day_id"]
        },
        coords={
            "time": windows_df["mid_time_hour"].values,
            "day_id": np.arange(stan_data["num_days"])
        }
    )
    
    idata.to_netcdf(output_dir / "inference_data_rain.nc")
    
    # Save auxiliary data
    df_raw.to_csv(output_dir / "merged_detector_data.csv", index=False)
    windows_df.to_csv(output_dir / "windowed_detector_data.csv", index=False)
    
    # UPDATED: Save B_diel for manual reconstruction in visualization
    np.savez(
        output_dir / "model_params.npz", 
        diel_step_min=params["diel_step_min"],
        burn_in_days=params["burn_in_days"],
        precip_daily=stan_data["precip_daily"],
        B_diel=stan_data["B_diel"]  # NEW: Required for reconstruction
    )
    
    # Print Summary 
    print("\n=== Parameter Summary ===")
    vars_to_summary = ["beta_0", "phi", "gamma_fast", "phi_fast", "gamma_slow", "phi_slow", "sigma_diel", "sigma_day"]
    print(az.summary(idata, var_names=vars_to_summary))
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()