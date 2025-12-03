#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import yaml


# ================================================================
# File discovery + loading
# ================================================================
def find_raven_tables(root_dir):
    raven_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                raven_files.append(os.path.join(root, f))
    return raven_files


def load_raven_table(path):
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception as e:
        print(f"WARNING: Could not parse {path}: {e}")
        return None


# ================================================================
# Metric extraction
# ================================================================
def extract_metrics(df):
    required = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")

    duration = df["End Time (s)"] - df["Begin Time (s)"]
    bandwidth = df["High Freq (Hz)"] - df["Low Freq (Hz)"]
    return duration.values, bandwidth.values


# ================================================================
# Main
# ================================================================
def main(root_dir, out_yaml="normalization.yaml"):
    raven_files = find_raven_tables(root_dir)
    print(f"Found {len(raven_files)} Raven tables.")

    durations = []
    bandwidths = []

    for path in raven_files:
        df = load_raven_table(path)
        if df is None:
            continue
        try:
            d, b = extract_metrics(df)
            durations.extend(d)
            bandwidths.extend(b)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    durations = np.array(durations, float)
    bandwidths = np.array(bandwidths, float)

    # --------------------------------------------------------------
    # Compute summary statistics
    # --------------------------------------------------------------
    stats = {
        "duration": {
            "mean": float(durations.mean()),
            "std": float(durations.std())
        },
        "bandwidth": {
            "mean": float(bandwidths.mean()),
            "std": float(bandwidths.std())
        }
    }

    # --------------------------------------------------------------
    # Save YAML
    # --------------------------------------------------------------
    with open(out_yaml, "w") as f:
        yaml.safe_dump(stats, f)

    print(f"Saved normalization YAML → {out_yaml}")
    print(stats)


if __name__ == "__main__":
    ANNOTATION_DIR = "/home/breallis/datasets/frog_calls/round_2"
    OUT = "normalization.yaml"
    main(ANNOTATION_DIR, OUT)
