### Visualizer script
#!/usr/bin/env python3
from __future__ import annotations

import ast
import json
import math
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import configs.nn_config as config


# -------------------------
# Parsing helpers
# -------------------------
def parse_array(x: Any) -> Optional[Any]:
    """Parse arrays or JSON/dict strings stored in CSV cells."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, (list, tuple)):
        return np.asarray(x)

    s = str(x).strip()
    if s == "":
        return None

    # Try JSON first (preferred)
    try:
        parsed = json.loads(s)
        return parsed
    except Exception:
        pass

    # Python literal (e.g. "[1,2,3]" or dict)
    try:
        parsed = ast.literal_eval(s)
        return parsed
    except Exception:
        pass

    # NumPy formatting (e.g. "[1.0 2.0 3.0]")
    try:
        return np.fromstring(s.replace("[", "").replace("]", ""), sep=" ")
    except Exception:
        return None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out


# -------------------------
# Exact PB convolution
# -------------------------
def pb_exact_convolution(slice_probs: np.ndarray) -> np.ndarray:
    """
    Exact distribution of sum of independent Bernoulli(slice_probs).
    Returns array of length n_slices+1 with probabilities for counts 0..n_slices.
    Uses iterative convolution for n_slices <= 256, otherwise pairwise FFT-based convolution.
    """
    p = np.asarray(slice_probs, dtype=float).flatten()
    n = len(p)
    if n == 0:
        return np.array([1.0])

    if n <= 256:
        probs = np.array([1.0], dtype=float)
        for pi in p:
            probs = np.convolve(probs, np.array([1.0 - pi, pi], dtype=float))
        return probs

    # Pairwise FFT convolution for larger n
    from numpy.fft import fft, ifft

    polys = deque([np.array([1.0 - pi, pi], dtype=float) for pi in p])
    while len(polys) > 1:
        a = polys.popleft()
        b = polys.popleft()
        size = len(a) + len(b) - 1
        nfft = 1 << (size - 1).bit_length()
        fa = fft(a, nfft)
        fb = fft(b, nfft)
        conv = np.real(ifft(fa * fb))[:size]
        polys.append(conv)
    return np.asarray(polys[0], dtype=float)


# -------------------------
# Events conversion heuristics
# -------------------------
def detect_and_convert_events(
    events_obj: Any,
    n_slices: int,
    sr: Optional[int] = None,
    clip_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse events object (dict or string) and convert starts/ends/conf to slice indices.
    Heuristics:
      - If values <= n_slices -> assume slice indices.
      - If values are small floats (<100) -> assume seconds.
      - If values are large -> assume sample indices.
    Returns (starts_idx, ends_idx, confs) as numpy arrays (floats).
    """
    if events_obj is None:
        return np.array([]), np.array([]), np.array([])

    if isinstance(events_obj, str):
        try:
            events_obj = json.loads(events_obj)
        except Exception:
            events_obj = parse_array(events_obj)

    if not isinstance(events_obj, dict):
        return np.array([]), np.array([]), np.array([])

    starts = np.asarray(events_obj.get("starts", []) or [], dtype=float)
    ends = np.asarray(events_obj.get("ends", []) or [], dtype=float)
    confs = np.asarray(events_obj.get("conf", []) or [], dtype=float)

    if starts.size == 0 or ends.size == 0:
        return np.array([]), np.array([]), np.array([])

    max_val = max(np.max(np.abs(starts)), np.max(np.abs(ends)))

    # Already slice indices
    if max_val <= n_slices:
        return starts, ends, confs

    # If sr and clip_seconds provided, use them to disambiguate
    if sr is not None and clip_seconds is not None:
        clip_samples = int(sr * clip_seconds)
        # If values look like sample indices
        if max_val > clip_samples * 0.5:
            slice_width_samples = clip_samples / float(n_slices)
            return starts / slice_width_samples, ends / slice_width_samples, confs
        # Otherwise treat as seconds
        slice_width_sec = clip_seconds / float(n_slices)
        return starts / slice_width_sec, ends / slice_width_sec, confs

    # Fallback heuristics without sr/clip_seconds
    if max_val < 100.0:
        # treat as seconds; need clip_seconds to convert, otherwise return raw
        if clip_seconds is None:
            return starts, ends, confs
        slice_width_sec = clip_seconds / float(n_slices)
        return starts / slice_width_sec, ends / slice_width_sec, confs

    # treat as sample indices but cannot convert without sr/clip_seconds
    return starts, ends, confs


# -------------------------
# Sampling helper
# -------------------------
def sample_by_gt(df: pd.DataFrame, n_per_bin: int = 3, seed: int = 42) -> pd.DataFrame:
    """Randomly sample a few examples from each rounded ground-truth count."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    if "gt_mu" not in df.columns:
        return df.sample(min(len(df), n_per_bin * 10), random_state=seed)
    df["gt_bin"] = np.rint(df["gt_mu"]).astype(int)
    sampled = []
    for _, group in df.groupby("gt_bin"):
        take = min(n_per_bin, len(group))
        sampled.append(group.sample(take, random_state=rng.integers(1_000_000)))
    return pd.concat(sampled).sort_index()



def plot_row(row: pd.Series, out_dir: str, idx: int, pb_exact: bool = True, pb_samples: int = 20000):
    # Parse ground truth and model outputs
    y_count = parse_array(row.get("q_k"))
    y_slice = parse_array(row.get("gt_slice"))

    # per-slice probabilities from count branch (we saved nn_count_slice_probs)
    count_slice_probs_raw = parse_array(row.get("nn_count_slice_probs"))
    # aggregated PMF over counts
    count_probs_raw = parse_array(row.get("nn_count_probs"))

    # events
    events_raw = row.get("events", None)

    dataset_idx = row.get("dataset_idx", None)
    n_slices_meta = row.get("n_slices", None)
    max_bin_meta = row.get("max_bin", None)

    # Basic validation
    if y_count is None or y_slice is None:
        print(f"[WARN] skipping sample {idx}: missing ground truth (q_k or gt_slice)")
        return

    if count_probs_raw is None:
        print(f"[WARN] skipping sample {idx}: missing nn_count_probs (aggregated PMF)")
        return

    # Convert to numpy arrays
    y_count = np.asarray(y_count, dtype=float).flatten()
    y_slice = np.asarray(y_slice, dtype=float).flatten()

    # count_slice_probs should already be probabilities (model applied sigmoid)
    count_slice_probs = None
    if count_slice_probs_raw is not None:
        count_slice_probs = np.asarray(count_slice_probs_raw, dtype=float).flatten()
        # defensive: if values outside [0,1], apply sigmoid
        if np.any(count_slice_probs < 0) or np.any(count_slice_probs > 1):
            count_slice_probs = _sigmoid(count_slice_probs)

    count_probs = np.asarray(count_probs_raw, dtype=float).flatten()

    # axes
    n_slices = len(y_slice)
    k_slices = np.arange(n_slices)
    k_count = np.arange(len(y_count))

    # To make step plots continue to the end, extend x and y by one point for where="post"
    k_slices_ext = np.arange(n_slices + 1)
    y_slice_ext = np.concatenate([y_slice, [y_slice[-1]]])
    count_slice_probs_ext = None
    if count_slice_probs is not None:
        if len(count_slice_probs) != n_slices:
            print(f"[WARN] count-slice length {len(count_slice_probs)} != n_slices {n_slices} for idx={idx}")
            minlen = min(len(count_slice_probs), n_slices)
            tmp = np.zeros(n_slices, dtype=float)
            tmp[:minlen] = count_slice_probs[:minlen]
            if minlen < n_slices:
                tmp[minlen:] = tmp[minlen - 1] if minlen > 0 else 0.0
            count_slice_probs_ext = np.concatenate([tmp, [tmp[-1]]])
        else:
            count_slice_probs_ext = np.concatenate([count_slice_probs, [count_slice_probs[-1]]])

    title_suffix = f" idx={idx}"
    if dataset_idx is not None and not (isinstance(dataset_idx, float) and math.isnan(dataset_idx)):
        title_suffix += f" (dataset_idx={int(dataset_idx)})"

    # FIGURE 1: probabilities comparison (aggregated count PMF shown for reference)
    plt.figure(figsize=(6, 4))
    # show aggregated PMF (count branch) and ground-truth PMF
    plt.bar(k_count, y_count, alpha=0.4, label="true")
    plt.plot(k_count, count_probs, lw=2, label="count branch PMF (aggregated)")
    plt.title("Count PMF comparison" + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{idx}_count_pmf.png"), bbox_inches="tight")
    plt.close()

    # FIGURE 2: intensity / per-slice comparison with events overlay (GT vs count-branch per-slice)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

    # events overlay (use original detect_and_convert_events behavior)
    sr = getattr(config, "DATASET_SAMPLE_RATE", None)
    clip_seconds = getattr(config, "CLIP_DURATION_SECONDS", None)
    n_slices_plot = int(n_slices_meta) if (n_slices_meta is not None and not (isinstance(n_slices_meta, float) and math.isnan(n_slices_meta))) else n_slices

    starts_idx, ends_idx, confs = detect_and_convert_events(events_raw, n_slices=n_slices_plot, sr=sr, clip_seconds=clip_seconds)
    if starts_idx.size > 0:
        for s, e, c in zip(starts_idx, ends_idx, confs):
            for ax in axes:
                ax.axvspan(s, e, alpha=float(c) * 0.3 if not math.isnan(float(c)) else 0.2, color="green", label=None)

    # Plot ground truth (extended so step continues to end)
    axes[0].step(k_slices_ext, y_slice_ext, where="post", color="black")
    axes[0].set_ylabel("ground truth")
    axes[0].set_ylim(-0.05, 1.05)

    # Plot count-branch per-slice probabilities (extended)
    if count_slice_probs_ext is not None:
        axes[1].step(k_slices_ext, count_slice_probs_ext, where="post", color="tab:orange")
    else:
        axes[1].text(0.5, 0.5, "no per-slice count-head outputs", ha="center", va="center")

    axes[1].set_ylabel("count-branch per-slice")
    axes[1].set_xlabel("slice index")
    axes[1].set_ylim(-0.05, 1.05)

    # Ensure x limits match slice axis exactly so spans and steps align and are not cropped
    for ax in axes:
        ax.set_xlim(0, n_slices)

    fig.suptitle(f"GT vs count-branch per-slice   gt_mu={row.get('gt_mu', float('nan')):.2f}")
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(os.path.join(out_dir, f"sample_{idx}_intensity.png"), bbox_inches="tight")
    plt.close()

    # FIGURE 3: PB comparison (slice-branch PB via exact convolution vs aggregated count PMF and true)
    # If we have no slice-branch per-slice probs, compute PB from count_slice_probs_ext if available,
    # otherwise skip slice-branch PB.
    if count_slice_probs is not None:
        # compute PB from per-slice count-branch probabilities (exact)
        slice_pb = pb_exact_convolution(count_slice_probs)
    else:
        # fallback: compute PB from slice_probs if you still have slice branch elsewhere (not used here)
        slice_pb = None

    pb_k = np.arange(len(slice_pb)) if slice_pb is not None else np.arange(len(count_probs))

    plt.figure(figsize=(6, 4))
    plt.bar(k_count, y_count, alpha=0.4, label="true")
    plt.plot(k_count, count_probs, lw=2, label="count branch PMF (aggregated)")
    if slice_pb is not None and slice_pb.sum() > 0:
        plt.plot(pb_k, slice_pb / np.sum(slice_pb), label="count-branch per-slice PB (exact)")
    plt.title("PB comparison" + title_suffix)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"sample_{idx}_pb.png"), bbox_inches="tight")
    plt.close()


# -------------------------
# Main
# -------------------------
def main(
    csv_path: Optional[str] = None,
    out_dir: Optional[str] = None,
    sample_per_gt: int = 3,
    pb_exact: bool = True,
    bin_by: str = "argmax_qk",  # options: "argmax_qk" or "round_mu"
):
    """
    Visualize samples and save images into subfolders per ground-truth bin.

    - If q_k is present, default behavior (argmax_qk) uses argmax(q_k) as the bin.
    - Otherwise falls back to rounding gt_mu.
    - Each sample's images are written to out_dir/gt_<bin>/sample_<idx>_*.png
    """
    if csv_path is None:
        csv_path = os.path.join(config.CHECKPOINT_DIR, "best.keras_multiband_calibration_full.csv")

    if out_dir is None:
        out_dir = os.path.join(config.CHECKPOINT_DIR, "nn_visualizations")

    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df_sampled = sample_by_gt(df, n_per_bin=sample_per_gt)

    print(f"[INFO] visualizing {len(df_sampled)} samples from {csv_path}")

    for idx, row in df_sampled.iterrows():
        try:
            # Determine ground-truth bin for this sample
            q_k_raw = parse_array(row.get("q_k"))
            gt_bin = None
            if q_k_raw is not None:
                q_k_arr = np.asarray(q_k_raw, dtype=float).flatten()
                if q_k_arr.size > 0:
                    if bin_by == "argmax_qk":
                        gt_bin = int(np.argmax(q_k_arr))
                    else:
                        # fallback to rounded mean if requested
                        gt_bin = int(round(np.sum(q_k_arr * np.arange(q_k_arr.size))))
            if gt_bin is None:
                # fallback to gt_mu if q_k missing or empty
                gt_mu = row.get("gt_mu", None)
                if gt_mu is not None and not (isinstance(gt_mu, float) and np.isnan(gt_mu)):
                    if bin_by == "round_mu":
                        gt_bin = int(round(float(gt_mu)))
                    else:
                        gt_bin = int(round(float(gt_mu)))
                else:
                    gt_bin = -1  # unknown bin

            # Create per-bin subfolder
            bin_folder_name = f"gt_{gt_bin}" if gt_bin >= 0 else "gt_unknown"
            out_dir_bin = os.path.join(out_dir, bin_folder_name)
            os.makedirs(out_dir_bin, exist_ok=True)

            # Call the plotting routine, writing files into the bin-specific folder
            plot_row(row, out_dir_bin, idx, pb_exact=pb_exact)

        except Exception as exc:
            print(f"[ERROR] failed plotting idx={idx}: {exc}")



if __name__ == "__main__":
    main()