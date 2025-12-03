import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
# Extract per-table metrics
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
# Normalization utilities
# ================================================================
def zscore(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-8)


def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


# ================================================================
# Combined metrics from (d, b) pairs
# ================================================================
def compute_combinations(d_norm, b_norm):
    """
    d_norm, b_norm should be arrays of identical length.
    Returns dict: name -> array.
    """
    d = np.asarray(d_norm, float)
    b = np.asarray(b_norm, float)

    combos = {}

    combos["arith_mean"] = (d + b) / 2
    combos["geom_mean"] = np.sqrt(np.abs(d * b))
    combos["harm_mean"] = 2 / (1/(np.abs(d)+1e-8) + 1/(np.abs(b)+1e-8))
    combos["euclidean_norm"] = np.sqrt(d**2 + b**2)
    combos["max"] = np.maximum(d, b)
    combos["min"] = np.minimum(d, b)
    combos["product"] = d * b

    return combos


# ================================================================
# Main analysis script
# ================================================================
def main(root_dir, out_prefix="raven_hist"):
    raven_files = find_raven_tables(root_dir)
    print(f"Found {len(raven_files)} Raven selection tables.")

    durations = []
    bandwidths = []

    # Collect data
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
    # Normalize
    # Choose: z-score for both
    # --------------------------------------------------------------
    d_norm = zscore(durations)
    b_norm = zscore(bandwidths)

    # If you want to switch to minmax normalization, replace with:
    # d_norm = minmax(durations)
    # b_norm = minmax(bandwidths)

    # Save histogram of normalized individual features
    plt.figure(figsize=(8,5))
    plt.hist(d_norm, bins=50)
    plt.title("Normalized Duration (z-score)")
    plt.xlabel("z-scored duration")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_duration_zscore.png")
    plt.close()

    plt.figure(figsize=(8,5))
    plt.hist(b_norm, bins=50)
    plt.title("Normalized Bandwidth (z-score)")
    plt.xlabel("z-scored bandwidth")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bandwidth_zscore.png")
    plt.close()

    # --------------------------------------------------------------
    # Combined metrics
    # --------------------------------------------------------------
    combos = compute_combinations(d_norm, b_norm)

    # Plot histogram for each combined metric
    for name, arr in combos.items():
        plt.figure(figsize=(8,5))
        plt.hist(arr, bins=50)
        plt.title(f"Combined metric: {name}")
        plt.xlabel(name)
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_{name}.png")
        plt.close()

    print("Done.")
    print("Saved:")
    print(f"- {out_prefix}_duration_zscore.png")
    print(f"- {out_prefix}_bandwidth_zscore.png")
    for name in combos.keys():
        print(f"- {out_prefix}_{name}.png")


if __name__ == "__main__":
    ANNOTATION_DIR = "/home/breallis/datasets/frog_calls/round_2"
    OUT = "hist"
    main(ANNOTATION_DIR, OUT)
