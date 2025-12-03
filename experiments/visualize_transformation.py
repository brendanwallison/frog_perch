import os, random
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import frog_perch.config as config
from frog_perch.utils.annotations import (
    annotation_confidence_from_features,
    get_frog_call_weights,
    soft_count_distribution
)

CONFIDENCE_PARAMS = config.CONFIDENCE_PARAMS

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

def extract_metrics(df):
    required = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'")
    duration = df["End Time (s)"] - df["Begin Time (s)"]
    bandwidth = df["High Freq (Hz)"] - df["Low Freq (Hz)"]
    return duration.values, bandwidth.values

# ================================================================
# Confidence histogram
# ================================================================
def make_confidence_hist(root_dir, out_prefix="conf_hist"):
    raven_files = find_raven_tables(root_dir)
    durations, bandwidths = [], []

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

    confs = []
    for d, b in zip(durations, bandwidths):
        c = annotation_confidence_from_features(
            d, b,
            duration_stats=CONFIDENCE_PARAMS["duration_stats"],
            bandwidth_stats=CONFIDENCE_PARAMS["bandwidth_stats"],
            k=CONFIDENCE_PARAMS["logistic_params"]["k"],
            x0=CONFIDENCE_PARAMS["logistic_params"]["x0"],
            lower=CONFIDENCE_PARAMS["logistic_params"]["lower"],
            upper=CONFIDENCE_PARAMS["logistic_params"]["upper"],
            clip_z=CONFIDENCE_PARAMS["logistic_params"]["clip_z"]
        )
        confs.append(c)

    confs = np.array(confs)

    plt.figure(figsize=(8,5))
    plt.hist(confs, bins=50, density=True, alpha=0.6, label="Histogram")
    try:
        import seaborn as sns
        sns.kdeplot(confs, color="red", label="KDE")
    except ImportError:
        pass
    for q in [0.25, 0.5, 0.75]:
        plt.axvline(np.quantile(confs, q), color="black", linestyle="--", alpha=0.7)
    plt.title("Distribution of transformed confidences")
    plt.xlabel("Confidence")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_confidence.png")
    plt.close()

# ================================================================
# Count shift analysis (windowed, with/without smoothing)
# ================================================================
def make_count_shift(root_dir, out_prefix="count_shift", sample_size=500, window=(0,5), top_bin_mean=4.5):
    raven_files = find_raven_tables(root_dir)
    hard_argmax, soft_argmax = [], []
    hard_E, soft_E = [], []

    clip_start, clip_end = window
    q2_soft = CONFIDENCE_PARAMS.get("q2_confidence", 0.75)

    for path in raven_files:
        df = load_raven_table(path)
        if df is None:
            continue
        if random.random() > 0.5:
            continue  # random subset

        # HARD counts: same weighting, smoothing disabled
        hard_weights = get_frog_call_weights(
            df,
            clip_start=clip_start,
            clip_end=clip_end,
            q2_confidence=1.0,
            use_continuous_confidence=False,
            duration_stats=CONFIDENCE_PARAMS["duration_stats"],
            bandwidth_stats=CONFIDENCE_PARAMS["bandwidth_stats"],
            logistic_params=CONFIDENCE_PARAMS["logistic_params"]
        )
        hard_dist = soft_count_distribution(hard_weights)
        hard_argmax.append(int(np.argmax(hard_dist)))
        hard_support = np.array([0, 1, 2, 3, top_bin_mean], dtype=float)
        hard_E.append(float(np.dot(hard_support, hard_dist)))

        # SOFT counts: confidence transformation enabled
        soft_weights = get_frog_call_weights(
            df,
            clip_start=clip_start,
            clip_end=clip_end,
            q2_confidence=q2_soft,
            use_continuous_confidence=True,
            duration_stats=CONFIDENCE_PARAMS["duration_stats"],
            bandwidth_stats=CONFIDENCE_PARAMS["bandwidth_stats"],
            logistic_params=CONFIDENCE_PARAMS["logistic_params"]
        )
        soft_dist = soft_count_distribution(soft_weights)
        soft_argmax.append(int(np.argmax(soft_dist)))
        soft_support = np.array([0, 1, 2, 3, top_bin_mean], dtype=float)
        soft_E.append(float(np.dot(soft_support, soft_dist)))

        if len(hard_argmax) >= sample_size:
            break

    # Histograms: argmax categories (0,1,2,3,4+)
    plt.figure(figsize=(10,5))
    bins = np.arange(0, 6) - 0.5  # centers at 0..5, where 4 represents 4+
    plt.hist(hard_argmax, bins=bins, alpha=0.6, label="Hard argmax (no smoothing)")
    plt.hist(soft_argmax, bins=bins, alpha=0.6, label="Soft argmax (conf transform)")
    plt.xticks([0,1,2,3,4], ["0","1","2","3","4+"])
    plt.title("Distribution of argmax counts (hard vs soft)")
    plt.xlabel("Count category")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_hist.png")
    plt.close()

    # Scatter: argmax shift
    plt.figure(figsize=(6,6))
    plt.scatter(hard_argmax, soft_argmax, alpha=0.5)
    plt.plot([0,4], [0,4], 'r--', label="Identity")
    plt.xticks([0,1,2,3,4], ["0","1","2","3","4+"])
    plt.yticks([0,1,2,3,4], ["0","1","2","3","4+"])
    plt.xlabel("Hard argmax (no smoothing)")
    plt.ylabel("Soft argmax (conf transform)")
    plt.title("Argmax shift between hard and soft")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_scatter.png")
    plt.close()

    # Expected count shift histograms
    plt.figure(figsize=(10,5))
    plt.hist(hard_E, bins=30, alpha=0.6, label="Hard expected count")
    plt.hist(soft_E, bins=30, alpha=0.6, label="Soft expected count")
    plt.title("Distribution of expected counts (hard vs soft)")
    plt.xlabel("Expected count (E[C])")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_expected_hist.png")
    plt.close()

    # Expected count shift scatter
    plt.figure(figsize=(6,6))
    plt.scatter(hard_E, soft_E, alpha=0.5)
    max_axis = max(hard_E + soft_E) if (hard_E and soft_E) else 4.0
    plt.plot([0, max_axis], [0, max_axis], 'r--', label="Identity")
    plt.xlabel("Hard expected count")
    plt.ylabel("Soft expected count")
    plt.title("Expected count shift (hard → soft)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_expected_scatter.png")
    plt.close()

# ================================================================
# Unified main
# ================================================================
def main(root_dir, out_prefix="analysis"):
    make_confidence_hist(root_dir, out_prefix+"_conf")
    make_count_shift(root_dir, out_prefix+"_count", sample_size=500, window=(0,5))
    print("All plots saved.")

if __name__ == "__main__":
    ANNOTATION_DIR = "/home/breallis/datasets/frog_calls/round_2"
    OUT_PREFIX = "hist"
    main(ANNOTATION_DIR, OUT_PREFIX)