"""
JAX/NumPyro SVI (MAP) pipeline for call-intensity hydrological decay model.
"""
import argparse
import shutil
import re
from pathlib import Path
import yaml
import json
import numpy as np
import pandas as pd

# Load Utilities
from frog_perch.stat_models.data.data_loading import load_detector_csvs
from frog_perch.stat_models.data.data_prep_for_rainfall_model import prepare_numpyro_data_hydrological

# Import the renamed MAP model function
from frog_perch.stat_models.numpyro.call_intensity_profile_rain_hill_map import compile_and_run

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_next_run_dir(base_output_dir):
    """Creates an incremented subfolder (e.g., run_001) to prevent overwriting."""
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

def main():
    parser = argparse.ArgumentParser(description="Run JAX Call-Intensity Rain Model (MAP Optimization)")
    
    default_config = Path("configs/call_intensity_splines_rainfall.yaml")
    
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
    climate_path = Path(cfg["climate_csv"])  # 🌟 Added: Parse climate file path
    calibration_json_path = Path(cfg["calibration_json_path"])
    
    # Setup safe execution directory
    run_dir = get_next_run_dir(cfg["output_dir"])
    print(f"📁 Target execution directory initialized: {run_dir}")
    
    shutil.copy2(args.config, run_dir / "run_config.yaml")
    
    # --- 1. Load & Prepare Data ---
    print(f"🚀 Loading detector data from {csv_dir}...")
    df_raw = load_detector_csvs(csv_dir)
    
    print(f"🌧️ Loading rainfall data from {rain_path}...")
    df_rain = pd.read_csv(rain_path)

    print(f"🌡️ Loading high-frequency climate log files from {climate_path}...")
    df_climate = pd.read_csv(climate_path)  # 🌟 Added: Read raw climate dataframe
    
    print(f"⚙️ Loading calibration parameters from {calibration_json_path}...")
    with open(calibration_json_path, 'r') as f:
        calib_params = json.load(f)
    
    print("🛠️ Preparing Hydrological Decay + Climate Alignment data...")
    numpyro_data, windows_df, params = prepare_numpyro_data_hydrological(
        df_raw,
        df_rain,
        df_climate,
        calibration_params=calib_params,
        knot_spacing_diel_min=cfg["knot_spacing_diel_min"],
        burn_in_days=cfg.get("burn_in_days", 14),
        w_fraction=cfg.get("w_fraction", 0.0167),
        output_dir=run_dir,
    )
    
    # --- 2. Run Inference (MAP Optimization) ---
    print("🔥 Starting Optimization (NumPyro - SVI/AutoDelta)...")
    svi_result, guide = compile_and_run(
        numpyro_data,
        num_steps=cfg.get("num_steps", 5000), 
        seed=cfg.get("seed", 0),
    )
    
    # --- 3. Save Results ---
    print(f"💾 Saving optimized parameters to {run_dir}...")
    
    # Extract the raw underlying numpy arrays from the JAX/NumPyro param dictionary
    map_params = {k: np.array(v) for k, v in svi_result.params.items()}
    
    # Save a single unified payload for the visualization script
    np.savez(
        run_dir / "map_predictions.npz", 
        losses=np.array(svi_result.losses),
        diel_step_min=params["diel_step_min"],
        burn_in_days=params["burn_in_days"],
        precip_daily=numpyro_data["precip_daily"],
        knots_grid=numpyro_data["knots_grid"], 
        temp_inter=numpyro_data["temp_inter"],
        temp_intra=numpyro_data["temp_intra"],
        rh_inter=numpyro_data["rh_inter"],
        rh_intra=numpyro_data["rh_intra"],
        light_inter=numpyro_data["light_inter"],
        light_intra=numpyro_data["light_intra"],
        **map_params
    )
    # Save auxiliary coordinate data
    df_raw.to_csv(run_dir / "merged_detector_data.csv", index=False)
    windows_df.to_csv(run_dir / "windowed_detector_data.csv", index=False)
    
    print("✅ Run complete.")

if __name__ == "__main__":
    main()