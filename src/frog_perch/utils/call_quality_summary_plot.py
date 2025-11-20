import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def find_raven_tables(root_dir):
    """Recursively find all .txt Raven annotation files."""
    raven_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".txt"):
                raven_files.append(os.path.join(root, f))
    return raven_files


def load_raven_table(path):
    """Load tab-delimited Raven selection table."""
    try:
        return pd.read_csv(path, sep="\t", engine="python")
    except Exception as e:
        print(f"WARNING: Could not parse {path}: {e}")
        return None


def extract_metrics(df):
    """
    Extract duration, bandwidth, and call quality category.
    Raven columns:
        Begin Time (s), End Time (s)
        Low Freq (Hz), High Freq (Hz)
        Annotation
    """
    required_cols = [
        "Begin Time (s)", "End Time (s)",
        "Low Freq (Hz)", "High Freq (Hz)",
        "Annotation"
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c}")

    durations = df["End Time (s)"] - df["Begin Time (s)"]
    bandwidths = df["High Freq (Hz)"] - df["Low Freq (Hz)"]

    annotations = df["Annotation"].astype(str).str.lower().fillna("")

    # Category: q2 vs clean
    category = np.where(annotations.str.contains("q2"), "q2", "clean")

    return durations.values, bandwidths.values, category


def aggregate_stats(durations, bandwidths, categories):
    """Compute per-class means and standard deviations."""
    stats = {}
    for cls in ["clean", "q2"]:
        mask = (categories == cls)
        stats[cls] = {
            "mean_duration": durations[mask].mean(),
            "std_duration": durations[mask].std(),
            "mean_bandwidth": bandwidths[mask].mean(),
            "std_bandwidth": bandwidths[mask].std(),
            "count": mask.sum(),
        }
    return stats


def main(root_dir, out="call_quality_bandwidth_plot.png"):
    raven_files = find_raven_tables(root_dir)
    print(f"Found {len(raven_files)} Raven selection tables.")

    all_durations = []
    all_bandwidths = []
    all_categories = []

    for path in raven_files:
        df = load_raven_table(path)
        if df is None:
            continue
        try:
            d, bw, c = extract_metrics(df)
            all_durations.extend(d)
            all_bandwidths.extend(bw)
            all_categories.extend(c)
        except Exception as e:
            print(f"Skipping {path}: {e}")

    durations = np.array(all_durations)
    bandwidths = np.array(all_bandwidths)
    categories = np.array(all_categories)

    stats = aggregate_stats(durations, bandwidths, categories)
    print("Stats:", stats)

    # --- Plotting ---
    plt.figure(figsize=(8, 6))

    colors = {"clean": "blue", "q2": "red"}

    for label in ["clean", "q2"]:
        s = stats[label]
        plt.errorbar(
            s["mean_bandwidth"],
            s["mean_duration"],
            xerr=s["std_bandwidth"],
            yerr=s["std_duration"],
            fmt="o", label=f"{label} (n={s['count']})",
            capsize=6, markersize=10, color=colors[label]
        )

    plt.xlabel("Bandwidth (Hz)")
    plt.ylabel("Duration (s)")
    plt.title("Call Duration vs Bandwidth by Annotation Quality")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()

    print(f"Saved plot to {out}")

if __name__ == "__main__":
    ANNOTATION_DIR = '/home/breallis/datasets/frog_calls/round_2' 
    OUT = "call_quality_mean_plot.png"
    main(ANNOTATION_DIR, OUT)
