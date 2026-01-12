#!/usr/bin/env python3
"""
Command-line script to run the full Stan call-intensity pipeline (HSGP Version).
"""

import argparse
from pathlib import Path
import yaml

from frog_perch.stat_models.inference.fit_call_intensity_windowed import (
    run_call_intensity_pipeline,
    print_diagnostics,
)

CONFIG_PATH = "/home/breallis/dev/frog_perch/src/frog_perch/stat_models/inference/call_intensity.yaml"

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
    parser = argparse.ArgumentParser(description="Run Stan call-intensity model (HSGP).")

    parser.add_argument("--csv-dir", type=Path, help="Directory containing detector CSV files.")
    parser.add_argument("--stan-model", type=Path, help="Path to call_intensity_hsgp.stan file.")
    parser.add_argument("--output-dir", type=Path, help="Directory to save results.")
    parser.add_argument("--config", type=Path, help="Optional YAML config file.")

    parser.add_argument("--warmup", type=int, help="Warmup iterations.")
    parser.add_argument("--sampling", type=int, help="Sampling iterations.")
    parser.add_argument("--chains", type=int, help="Number of MCMC chains.")
    parser.add_argument("--seed", type=int, help="Random seed.")
    
    # --- HSGP Frequencies (Replaces Spline Knots) ---
    parser.add_argument("--m-season", type=int, dest="M_season", help="Number of seasonal frequencies (sine/cos pairs).")
    parser.add_argument("--m-diel", type=int, dest="M_diel", help="Number of diel frequencies (sine/cos pairs).")

    parser.add_argument("--a-call", type=float, help="Beta(a, b) for calls.")
    parser.add_argument("--b-call", type=float, help="Beta(a, b) for calls.")
    parser.add_argument("--a-bg", type=float, help="Beta(a, b) for bg.")
    parser.add_argument("--b-bg", type=float, help="Beta(a, b) for bg.")

    args = parser.parse_args()

    defaults = {
        "warmup": 1000,
        "sampling": 1000,
        "chains": 4,
        "seed": 12345,
        "use_binning": True,
        
        # Default Frequencies (M=10 means 20 basis functions)
        "M_season": 10,
        "M_diel": 10,

        "a_call": 3.0,
        "b_call": 1.0,
        "a_bg": 0.5,
        "b_bg": 10.0,
    }

    config_path = args.config if args.config is not None else CONFIG_PATH
    config_file = load_config_file(Path(config_path))
    cli_dict = vars(args)
    cfg = merge_configs(defaults, config_file, cli_dict)

    for key in ["csv_dir", "stan_model", "output_dir"]:
        if isinstance(cfg.get(key), str):
            cfg[key] = Path(cfg[key])

    required = ["csv_dir", "stan_model", "output_dir"]
    missing = [r for r in required if cfg.get(r) is None]
    if missing:
        raise ValueError(f"Missing required configuration values: {missing}")

    cfg["output_dir"].mkdir(parents=True, exist_ok=True)

    print("Running call-intensity HSGP pipeline with configuration:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")

    # Pass the M arguments to the pipeline
    # Note: Ensure fit_call_intensity.py is updated to accept M_season/M_diel
    df, fit = run_call_intensity_pipeline(
        csv_dir=cfg["csv_dir"],
        stan_model_path=cfg["stan_model"],
        iter_warmup=cfg["warmup"],
        iter_sampling=cfg["sampling"],
        chains=cfg["chains"],
        seed=cfg["seed"],
        a_call=cfg["a_call"],
        b_call=cfg["b_call"],
        a_bg=cfg["a_bg"],
        b_bg=cfg["b_bg"],
        use_binning=True, 
        
        # Updated Args
        K_season=cfg["K_season"],
        K_diel=cfg["K_diel"],
    )

    df.to_csv(cfg["output_dir"] / "merged_detector_data.csv", index=False)
    fit.save_csvfiles(cfg["output_dir"])
    print_diagnostics(fit)
    print("\n✅ Finished successfully.")

if __name__ == "__main__":
    main()