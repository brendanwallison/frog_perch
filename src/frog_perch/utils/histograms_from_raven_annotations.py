import os
import pandas as pd
import matplotlib.pyplot as plt

def find_raven_tables(root_dir):
    """
    Recursively find all .txt files that look like Raven selection tables.
    """
    raven_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                raven_files.append(os.path.join(root, f))
    return raven_files


def load_raven_table(path):
    """
    Load a Raven selection table using pandas.
    Raven tables are tab-delimited, usually with headers including:
    ['Selection', 'View', 'Channel', 'Begin Time (s)', 'End Time (s)',
     'Low Freq (Hz)', 'High Freq (Hz)', ...]
    """
    try:
        df = pd.read_csv(path, sep="\t", engine="python")
        return df
    except Exception as e:
        print(f"WARNING: Could not parse {path}: {e}")
        return None


def extract_metrics(df):
    """
    Extract duration and bandwidth arrays from a Raven selection table dataframe.
    """
    required_cols = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in selection table!")

    duration = df["End Time (s)"] - df["Begin Time (s)"]
    bandwidth = df["High Freq (Hz)"] - df["Low Freq (Hz)"]

    return duration.values, bandwidth.values


def main(root_dir, out_prefix="raven_hist"):
    raven_files = find_raven_tables(root_dir)
    print(f"Found {len(raven_files)} Raven selection tables.")

    all_durations = []
    all_bandwidths = []

    for path in raven_files:
        df = load_raven_table(path)
        if df is None:
            continue
        try:
            duration, bandwidth = extract_metrics(df)
            all_durations.extend(duration)
            all_bandwidths.extend(bandwidth)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    # --- Plot Duration Histogram ---
    plt.figure(figsize=(8,5))
    plt.hist(all_durations, bins=50)
    plt.xlabel("Duration (s)")
    plt.ylabel("Count")
    plt.title("Histogram of Call Durations")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_duration.png", dpi=150)
    plt.close()

    # --- Plot Bandwidth Histogram ---
    plt.figure(figsize=(8,5))
    plt.hist(all_bandwidths, bins=50)
    plt.xlabel("Bandwidth (Hz)")
    plt.ylabel("Count")
    plt.title("Histogram of Call Bandwidths")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_bandwidth.png", dpi=150)
    plt.close()

    print(f"Saved histograms to {out_prefix}_duration.png and {out_prefix}_bandwidth.png")


if __name__ == "__main__":
    ANNOTATION_DIR = '/home/breallis/datasets/frog_calls/round_2' 
    OUT = "hist"
    main(ANNOTATION_DIR, OUT)
