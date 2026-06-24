#!/usr/bin/env python3
"""
Pipeline runner script for call-intensity hydrological lag model.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import yaml
import json
import numpy as np
import arviz as az
import pandas as pd

from frog_perch.stat_models.data.data_loading import load_detector_csvs
from frog_perch.stat_models.data.data_prep_for_rainfall_model import prepare_numpyro_data_hydrological

# Explicit clean import from your isolated model file
from frog_perch.stat_models.numpyro.call_intensity_profile_rain_hill import compile_and_run

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Run JAX Call-Intensity Rain Model")
    default_config = Path("configs/call_intensity_splines_rainfall.yaml")
    parser.add_argument("--config", type=Path, default=default_config)
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, output_dir / "run_config.yaml")
    
    df_raw = load_detector_csvs(Path(cfg["csv_dir"]))
    df_rain = pd.read_csv(Path(cfg["rain_csv"]))
    df_climate = pd.read_csv(cfg["climate_csv"])
    with open(Path(cfg["calibration_json_path"]), 'r') as f:
        calib_params = json.load(f)
    
    print("🛠️ Preparing Hydrological Decay data...")
    numpyro_data, windows_df, params = prepare_numpyro_data_hydrological(
        df_raw, df_rain, df_climate,
        calibration_params=calib_params,
        knot_spacing_diel_min=cfg["knot_spacing_diel_min"],
        burn_in_days=cfg.get("burn_in_days", 14),
        w_fraction=cfg.get("w_fraction", 0.0167),
        output_dir=output_dir
    )
    
    print("🔥 Starting Parallel NUTS Chains (NumPyro Baseline Model)...")
    mcmc = compile_and_run(
        stan_data=numpyro_data,
        num_warmup=int(cfg["warmup"]),
        num_samples=int(cfg["sampling"]),
        num_chains=int(cfg["chains"]),
        seed=int(cfg.get("seed", 0))
    )
    
    print(f"💾 Converting to ArviZ InferenceData and saving to {output_dir}...")
    idata = az.from_numpyro(
        mcmc,
        constant_data={
            "w_obs": numpyro_data["w_obs"],
            "precip_daily": numpyro_data["precip_daily"],
            "temp": numpyro_data["temp"],
            "rh": numpyro_data["rh"],
            "rms_obs": numpyro_data["rms_obs"]
        },
        dims={
            "alpha_day_raw": ["day_id"],
            "alpha_day": ["day_id"]
        },
        coords={
            "time": windows_df["mid_time_hour"].values,
            "day_id": np.arange(numpyro_data["num_days"])
        }
    )
    idata.to_netcdf(output_dir / "inference_data_rain.nc")
    
    windows_df.to_csv(output_dir / "windowed_detector_data.csv", index=False)
    
    np.savez(
        output_dir / "model_params.npz",
        burn_in_days=params["burn_in_days"],
        precip_daily=numpyro_data["precip_daily"],
        knots_grid=numpyro_data["knots_grid"],
        time_of_day=numpyro_data["time_of_day"],
        temp=numpyro_data["temp"],
        rh=numpyro_data["rh"],
        rms_obs=numpyro_data["rms_obs"]
    )
        
    # --- Aligned ArviZ Summary Variable Target Block ---
    print("\n=== Parameter Convergence & Summary ===")
    vars_to_summary = [
        "beta_0",
        "phi",
        "b_p0",
        "b_p1",
        "b_p2",
        "half_life_slow_val",
        "tau_pool",
        "b_shape",
        "gamma_plateau",
        "b_rms_val",

        "b_temp_val",
        "b_rh_val",

        "sigma_diel",
        "b_day_val",
        "delta_seasonal_val",
    ]
    
    available_vars = [
        v for v in vars_to_summary
        if hasattr(idata, "posterior") and v in idata.posterior.data_vars
    ]
        
    summary_df = az.summary(idata, var_names=available_vars)
    print(summary_df)
    
    # 🌟 FIX: Force cast the pandas series extraction to a float
    max_rhat = float(summary_df["r_hat"].max())
    print(f"\n📈 Maximum R-hat value across core tracking blocks: {max_rhat:.4f}")
    if max_rhat > 1.05:
        print("⚠️ WARNING: High R-hat detected. Chains may not have mixed completely.")
        
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()