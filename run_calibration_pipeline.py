#!/usr/bin/env python3
"""
run_calibration_pipeline.py

Orchestrates the two-step Bayesian calibration process:
1. Extracts tracking features (moments, counts, covariates) from the P2 holdout split.
2. Optimizes the sensor model parameters to fit the extracted dataset.
"""
import os
import argparse
import yaml
from pathlib import Path

# Import the global config
import configs.nn_config as config

# Import our refactored endpoints
from frog_perch.nn_calibration.calibration_feature_extraction import extract_calibration_features
from frog_perch.nn_calibration.calibrate import run_calibration

def get_config_dict(cfg_module) -> dict:
    """Converts all uppercase variables in a python module to a dictionary."""
    return {k: getattr(cfg_module, k) for k in dir(cfg_module) if k.isupper()}

def load_normalization_stats(yaml_path: str) -> dict:
    """Safely load the generated normalization stats."""
    try:
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Could not find {yaml_path}. Have you run the normalization script yet?"
        )

def main():
    parser = argparse.ArgumentParser(description="Extract features and calibrate the sensor model.")
    
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="best.keras", 
        help="Filename of the model checkpoint (inside config.CHECKPOINT_DIR). Default: best.keras"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test", 
        help="Dataset split to use for calibration extraction. Default: test"
    )
    
    args = parser.parse_args()
    config_dict = get_config_dict(config)

    # --- MISSING PARITY LOGIC RESTORED HERE ---
    config_dir = Path(config.__file__).parent
    norm_stats = load_normalization_stats(config_dir / "normalization.yaml")
    
    config_dict["CONFIDENCE_PARAMS"] = {
        "duration_stats": norm_stats.get("duration", {}),
        "bandwidth_stats": norm_stats.get("bandwidth", {}),
        "logistic_params": config_dict.get("CONFIDENCE_LOGISTIC_PARAMS", {})
    }
    # ------------------------------------------

    print("\n=======================================================")
    print("      BAYESIAN SENSOR MODEL CALIBRATION PIPELINE       ")
    print("=======================================================")
    print(f"[CONFIG] Checkpoint:    {args.ckpt}")
    print(f"[CONFIG] Target Split:  {args.split}")
    print(f"[CONFIG] Max Bin (K):   {config_dict.get('MAX_BIN', 8)}\n")

    # ---------------------------------------------------------
    # STEP 1: Feature Extraction
    # ---------------------------------------------------------
    print("--- STEP 1: Extracting Calibration Features ---")
    
    extract_calibration_features(
        config_dict=config_dict, 
        ckpt_name=args.ckpt, 
        split=args.split
    )
    
    ckpt_dir = config_dict.get("CHECKPOINT_DIR", "")
    csv_path = os.path.join(ckpt_dir, f"{args.ckpt}_multiband_calibration.csv")

    if not os.path.exists(csv_path):
        print(f"\n[FATAL ERROR] Extraction failed to produce CSV at: {csv_path}")
        return

    # ---------------------------------------------------------
    # STEP 2: Bayesian Optimization
    # ---------------------------------------------------------
    print("\n--- STEP 2: Optimizing Sensor Model Parameters ---")
    
    fitted_params = run_calibration(
        config_dict=config_dict, 
        csv_path=csv_path
    )
    
    if fitted_params:
        print("\n[SUCCESS] Calibration Pipeline Completed.")
    else:
        print("\n[FAILURE] Pipeline halted due to optimization failure.")

if __name__ == "__main__":
    main()