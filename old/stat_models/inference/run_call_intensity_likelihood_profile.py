#!/usr/bin/env python3
"""
Command-line script to run the full Stan call-intensity pipeline
(Likelihood Profile + HSGP Version).
"""

import argparse
from pathlib import Path
import yaml
import numpy as np

# Ensure this import matches your actual file name
from frog_perch.stat_models.inference.fit_call_intensity_likelihood_profile import (
    run_call_intensity_pipeline,
    print_diagnostics,
)

# You can update this default path if you have a new config file location
CONFIG_PATH = "/home/breallis/dev/frog_perch/src/frog_perch/stat_models/inference/call_intensity_likelihood_profile.yaml"

def load_config_file(path: Path) -> dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}

def merge_configs(defaults: dict, config_file: dict, cli_args: dict) -> dict:
    merged = defaults.copy()
    merged.update(config_file)
    for k, v in cli_args.items():
        if v is not None:
            merged[k] = v
    return merged

def main():
    parser = argparse.ArgumentParser(description="Run Stan call-intensity model (Likelihood Profile + HSGP).")

    # --- IO Paths ---
    parser.add_argument("--csv-dir", type=Path, help="Directory containing detector CSV files.")
    parser.add_argument("--stan-model", type=Path, help="Path to call_intensity_profile.stan file.")
    parser.add_argument("--output-dir", type=Path, help="Directory to save results.")
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")
    parser.add_argument("--cache-file", type=str, help="Optional path to pickle cache for preprocessed data.")

    # --- MCMC Settings ---
    parser.add_argument("--warmup", type=int, help="Warmup iterations.")
    parser.add_argument("--sampling", type=int, help="Sampling iterations.")
    parser.add_argument("--chains", type=int, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    
    # --- HSGP Frequencies ---
    parser.add_argument("--m-season", type=int, dest="M_season", help="Number of seasonal frequencies (sine/cos pairs).")
    parser.add_argument("--m-diel", type=int, dest="M_diel", help="Number of diel frequencies (sine/cos pairs).")

    # --- Aggregation / Windowing ---
    parser.add_argument("--bin-duration", type=float, dest="bin_duration_sec", help="Sub-second bin duration (e.g. 0.2s).")
    parser.add_argument("--window-length", type=float, dest="window_length_sec", help="Analysis window length (e.g. 5.0s).")

    # --- Calibration (Beta Params) ---
    parser.add_argument("--a-call", type=float, help="Beta(a, b) for calls.")
    parser.add_argument("--b-call", type=float, help="Beta(a, b) for calls.")
    parser.add_argument("--a-bg", type=float, help="Beta(a, b) for bg.")
    parser.add_argument("--b-bg", type=float, help="Beta(a, b) for bg.")

    args = parser.parse_args()

    # --- Defaults ---
    defaults = {
        "warmup": 1000,
        "sampling": 1000,
        "chains": 4,
        "seed": 12345,
        
        # Aggregation Defaults
        "bin_duration_sec": 0.2,   # 200ms bins
        "window_length_sec": 5.0,  # 5s analysis window
        
        # Default Frequencies (M=10 means 20 basis functions)
        "M_season": 10,
        "M_diel": 10,

        "a_call": 3.0,
        "b_call": 1.0,
        "a_bg": 1.0,
        "b_bg": 5.0,
    }

    config_path = args.config if args.config is not None else CONFIG_PATH
    if config_path:
        config_path = Path(config_path) # ensure Path object if string passed
        
    config_file = load_config_file(config_path)
    cli_dict = vars(args)
    cfg = merge_configs(defaults, config_file, cli_dict)

    # Convert paths to Path objects
    for key in ["csv_dir", "stan_model", "output_dir"]:
        if isinstance(cfg.get(key), str):
            cfg[key] = Path(cfg[key])

    required = ["csv_dir", "stan_model", "output_dir"]
    missing = [r for r in required if cfg.get(r) is None]
    if missing:
        raise ValueError(f"Missing required configuration values: {missing}")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    print("Running Call-Intensity Likelihood Profile Pipeline with configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # --- Run Pipeline ---
    # Note: Ensure the function returns 4 items now (df, metadata, hsgp_params, fit)
    df, window_metadata, hsgp_params, fit = run_call_intensity_pipeline(
        csv_dir=cfg["csv_dir"],
        stan_model_path=cfg["stan_model"],
        
        # MCMC
        iter_warmup=cfg["warmup"],
        iter_sampling=cfg["sampling"],
        chains=cfg["chains"],
        seed=cfg["seed"],
        
        # Calibration
        a_call=cfg["a_call"],
        b_call=cfg["b_call"],
        a_bg=cfg["a_bg"],
        b_bg=cfg["b_bg"],
        
        # Aggregation
        bin_duration_sec=cfg["bin_duration_sec"],
        window_length_sec=cfg["window_length_sec"],
        
        # HSGP
        M_season=cfg["M_season"],
        M_diel=cfg["M_diel"],
        
        # Caching
        cache_file=cfg.get("cache_file"),
    )

    # Save outputs
    print(f"\nSaving results to {cfg['output_dir']}...")
    
    # 1. Save DataFrames
    df.to_csv(cfg["output_dir"] / "merged_detector_data.csv", index=False)
    window_metadata.to_csv(cfg["output_dir"] / "windowed_detector_data.csv", index=False)
    
    # 2. Save HSGP Parameters to .npz (simpler than JSON for numpy types)
    # When loading later: data = np.load(..., allow_pickle=True)
    np.savez(
        cfg["output_dir"] / "hsgp_params.npz",
        season=hsgp_params["season"],
        diel=hsgp_params["diel"]
    )
    
    # 3. Save Stan CSV files (chains)
    fit.save_csvfiles(dir=str(cfg["output_dir"]))
    
    print_diagnostics(fit)
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()