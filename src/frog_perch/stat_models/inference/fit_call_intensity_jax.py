"""
JAX/NumPyro pipeline for call-intensity model.
"""
import argparse
from pathlib import Path
import yaml
import numpy as np
import arviz as az
import pandas as pd

# 1. Load Data Loading Utility
from frog_perch.stat_models.inference.data_loading import load_detector_csvs

# 2. Load Data Prep Utility 
from frog_perch.stat_models.inference.prepare_stan_data_likelihood_profile_splines import prepare_stan_data_splines

# 3. Load JAX Model
from frog_perch.stat_models.numpyro.call_intensity_profile_spline import compile_and_run

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run JAX Call-Intensity Model")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Convert paths
    csv_dir = Path(cfg["csv_dir"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # --- 1. Load & Prepare Data ---
    print(f"Loading data from {csv_dir}...")
    df_raw = load_detector_csvs(csv_dir)
    
    print("Preparing Likelihood Profile data (Splines)...")
    stan_data, windows_df, spline_params = prepare_stan_data_splines(
        df_raw,
        a_call=cfg["a_call"], 
        b_call=cfg["b_call"],
        a_bg=cfg["a_bg"],     
        b_bg=cfg["b_bg"],
        bin_duration_sec=cfg["bin_duration_sec"],
        window_length_sec=cfg["window_length_sec"],
        knot_spacing_season_days=cfg["knot_spacing_season_days"],
        knot_spacing_diel_min=cfg["knot_spacing_diel_min"]
    )
    
    # Verify shapes
    print(f"  > Windows (T): {stan_data['T']}")
    print(f"  > Season Basis Cols (K): {stan_data['K_season']}")
    print(f"  > Diel Basis Cols (K): {stan_data['K_diel']}")
    
    # --- 2. Run Inference (JAX) ---
    print("Starting Sampling (NumPyro)...")
    mcmc = compile_and_run(
        stan_data,
        num_warmup=cfg["warmup"],
        num_samples=cfg["sampling"],
        num_chains=cfg["chains"],
        seed=cfg["seed"]
    )
    
    # --- 3. Save Results ---
    print(f"Saving results to {output_dir}...")
    
    # Convert to ArviZ InferenceData
    idata = az.from_numpyro(
        mcmc,
        constant_data={
            "w_obs": stan_data["w_obs"],
            "B_season": stan_data["B_season"]
        },
        dims={
            "trend_season": ["time"],
            "trend_diel": ["time"],
            "lambda": ["time"],
            # CHANGE 1: Added alpha_day dimension, Removed eps
            "alpha_day": ["day_id"]
        },
        coords={
            "time": windows_df["start_time"].values,
            # We don't strictly need to label day_id here, 
            # ArviZ will just number them 0..D-1, which matches our day_idx
        }
    )
    
    # Save InferenceData (NetCDF - Standard for JAX)
    idata.to_netcdf(output_dir / "inference_data.nc")
    
    # Save CSVs (Metadata & Raw) to match Stan pipeline outputs
    df_raw.to_csv(output_dir / "merged_detector_data.csv", index=False)
    windows_df.to_csv(output_dir / "windowed_detector_data.csv", index=False)
    
    # Save Spline Parameters
    np.savez(
        output_dir / "spline_params.npz", 
        season_step=spline_params["season_step"],
        diel_step_min=spline_params["diel_step_min"],
        diel_bounds=spline_params["diel_bounds"]
    )
    
    # Print Summary 
    # CHANGE 2: Removed sigma_proc, Added sigma_day
    print("\n=== Parameter Summary ===")
    print(az.summary(idata, var_names=["beta_0", "phi", "sigma_season", "sigma_diel", "sigma_day"]))
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()