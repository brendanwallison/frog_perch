#!/usr/bin/env python3
"""
run_feature_extraction.py

Execution script to run the bulk feature extraction pipeline for NN calibration.
Loads the global configuration, parses command-line overrides, and passes 
everything to the processing module.
"""
import argparse

# Import the global config
import configs.nn_config as config

# Import our refactored module
from src.frog_perch.nn_calibration.feature_extraction_pipeline import process_directory

def get_config_dict(cfg_module) -> dict:
    """Converts all uppercase variables in a python module to a dictionary."""
    return {k: getattr(cfg_module, k) for k in dir(cfg_module) if k.isupper()}

def main():
    # Setup default paths (you can leave these blank and make them required, 
    # but providing defaults for your specific machine speeds up daily testing)
    default_input = "/home/breallis/datasets/frog_calls/gabon_full/P2"
    default_output = "/home/breallis/datasets/frog_calls/gabon_full/P2_nn_features"

    parser = argparse.ArgumentParser(description="Run bulk feature extraction for field audio.")
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        default=default_input,
        help="Path to the directory containing raw field audio."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=default_output,
        help="Path to save the generated CSV files."
    )
    parser.add_argument(
        "--ckpt", 
        type=str, 
        default="best.keras", 
        help="Filename of the model checkpoint (inside config.CHECKPOINT_DIR). Default: best.keras"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for Perch/Keras inference."
    )
    parser.add_argument(
        "--step_seconds", 
        type=float, 
        default=5.0,
        help="Window stride in seconds. Default is 5.0."
    )
    parser.add_argument(
        "--overwrite", 
        action="store_true", 
        help="Overwrite existing output CSV files."
    )

    args = parser.parse_args()

    # Build the config dictionary dynamically from the central config file
    config_dict = get_config_dict(config)

    print("\n[CONFIG] Loaded global config into dictionary.")
    print(f"[CONFIG] Input Directory:  {args.input_dir}")
    print(f"[CONFIG] Output Directory: {args.output_dir}")
    print(f"[CONFIG] Checkpoint File:  {args.ckpt}\n")

    # Execute the pipeline
    process_directory(
        config_dict=config_dict,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        ckpt_filename=args.ckpt,
        batch_size=args.batch_size,
        step_seconds=args.step_seconds,
        overwrite=args.overwrite
    )
    
    print("\n[INFO] Feature extraction pipeline finished successfully.")

if __name__ == "__main__":
    main()