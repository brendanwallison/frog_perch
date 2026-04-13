import os
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------------------------------------------
# I/O layer (pandas isolated here only)
# ------------------------------------------------------------
def load_annotations(annotation_dir: str, audio_file: str) -> pd.DataFrame:
    """
    Load Raven annotation table for a single audio file.

    This is the ONLY function allowed to use pandas.

    Returns a normalized dataframe with lowercase annotation labels.
    """
    base = os.path.splitext(audio_file)[0]
    path = os.path.join(annotation_dir, f"{base}.Table.1.selections.txt")

    if not os.path.exists(path):
        return pd.DataFrame(columns=["Annotation", "Begin Time (s)", "End Time (s)"])

    df = pd.read_csv(path, sep="\t")
    df["Annotation"] = df["Annotation"].astype(str).str.strip().str.lower()
    return df


# ------------------------------------------------------------
# Event extraction (cached per audio file)
# ------------------------------------------------------------
def _df_to_events(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert dataframe → numpy event representation.

    Returns:
        starts: float32 array
        ends: float32 array
        bandwidths: float32 array
    """
    if df.empty:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    starts = pd.to_numeric(df["Begin Time (s)"], errors="coerce").to_numpy(dtype=np.float32)
    ends = pd.to_numeric(df["End Time (s)"], errors="coerce").to_numpy(dtype=np.float32)

    # Bandwidth is optional but needed for confidence model
    if "Bandwidth (Hz)" in df.columns:
        bandwidths = pd.to_numeric(df["Bandwidth (Hz)"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        bandwidths = np.zeros_like(starts, dtype=np.float32)

    valid = (
        ~np.isnan(starts)
        & ~np.isnan(ends)
        & (ends > starts)
    )

    return starts[valid], ends[valid], bandwidths[valid]


@lru_cache(maxsize=512)
def get_event_cache(annotation_dir: str, audio_file: str):
    """
    Cached event table for an audio file.

    This removes ALL repeated pandas + disk IO in training.
    """
    df = load_annotations(annotation_dir, audio_file)
    return _df_to_events(df)


# ------------------------------------------------------------
# Confidence model (pure NumPy, reused everywhere)
# ------------------------------------------------------------
def compute_event_confidence(
    durations: np.ndarray,
    bandwidths: np.ndarray,
    duration_stats: Dict[str, float],
    bandwidth_stats: Dict[str, float],
    combine=(0.5, 0.5),
    k: float = 1.0,
    x0: float = -3.0,
    lower: float = 0.01,
    upper: float = 0.99,
    clip_z: float = 5.0,
) -> np.ndarray:
    """
    Logistic confidence model over standardized event features.

    Output is per-event confidence multiplier in [lower, upper].
    """
    d_mean, d_std = duration_stats["mean"], max(1e-6, duration_stats["std"])
    b_mean, b_std = bandwidth_stats["mean"], max(1e-6, bandwidth_stats["std"])

    z_d = np.clip((durations - d_mean) / d_std, -clip_z, clip_z)
    z_b = np.clip((bandwidths - b_mean) / b_std, -clip_z, clip_z)

    w1, w2 = combine if isinstance(combine, (tuple, list)) else (0.5, 0.5)
    z = w1 * z_d + w2 * z_b

    sig = 1.0 / (1.0 + np.exp(-k * (z - x0)))
    return np.clip(lower + (upper - lower) * sig, lower, upper).astype(np.float32)

def compute_window_overlap(
    starts: np.ndarray,
    ends: np.ndarray,
    clip_start: float,
    clip_end: float,
) -> np.ndarray:
    """
    Fractional overlap PER EVENT (no filtering).
    Output always aligned with input event arrays.
    """

    if len(starts) == 0 or clip_end <= clip_start:
        return np.zeros_like(starts, dtype=np.float32)

    overlap = np.minimum(clip_end, ends) - np.maximum(clip_start, starts)
    duration = ends - starts

    # IMPORTANT: NO masking, NO compression
    frac = overlap / np.maximum(duration, 1e-12)

    # clamp to valid range
    frac = np.clip(frac, 0.0, 1.0)

    return frac.astype(np.float32)


def compute_slice_overlap_matrix(
    starts: np.ndarray,
    ends: np.ndarray,
    clip_start: float,
    clip_end: float,
    n_slices: int,
) -> np.ndarray:
    """
    Vectorized slice-event overlap matrix.

    Returns:
        shape = (n_events, n_slices)

    Each entry is:
        event_overlap_fraction within that slice

    IMPORTANT:
    This is different from window overlap because:
    - window = full clip
    - slice = sub-interval partition
    """
    if len(starts) == 0 or n_slices <= 0 or clip_end <= clip_start:
        return np.zeros((0, n_slices), dtype=np.float32)

    # slice boundaries
    edges = np.linspace(clip_start, clip_end, n_slices + 1, dtype=np.float32)

    # expand to (n_events, n_slices)
    s = starts[:, None]
    e = ends[:, None]

    s2 = edges[:-1][None, :]
    e2 = edges[1:][None, :]

    overlap = np.minimum(e, e2) - np.maximum(s, s2)
    duration = np.maximum(e - s, 1e-12)

    frac = np.clip(overlap / duration, 0.0, 1.0)

    return frac.astype(np.float32)


# ------------------------------------------------------------
# Poisson-binomial views
# ------------------------------------------------------------
def binary_clip_probability(p: np.ndarray) -> float:
    """
    P(at least one event) = 1 - Π(1 - p_i)
    """
    if p is None or len(p) == 0:
        return 0.0
    p = np.asarray(p, dtype=np.float32)
    return float(1.0 - np.prod(1.0 - p))


def soft_count_distribution(weights: np.ndarray, max_bin: int = 16) -> np.ndarray:
    """
    Poisson-binomial distribution over event counts.

    weights are per-event probabilities.
    """
    max_bin = int(max_bin)

    if weights is None or len(weights) == 0:
        out = np.zeros(max_bin + 1, dtype=np.float32)
        out[0] = 1.0
        return out

    dist = np.zeros(max_bin + 1, dtype=np.float32)
    dist[0] = 1.0

    for p in np.clip(weights, 0.0, 1.0):
        if p == 0:
            continue

        new = np.zeros_like(dist)
        new += dist * (1.0 - p)
        new[1:] += dist[:-1] * p
        new[-1] += dist[-1] * p
        dist = new

    s = dist.sum()
    if s > 0:
        dist /= s
    else:
        dist[:] = 0
        dist[0] = 1.0

    return dist.astype(np.float32)


# ------------------------------------------------------------
# Slice aggregation (vectorized, no loops over events)
# ------------------------------------------------------------
def slice_binary_confidences(
    slice_overlap_matrix: np.ndarray,
) -> np.ndarray:
    """
    Aggregate slice probabilities using independent-event assumption.

    Input:
        (n_events, n_slices)

    Output:
        (n_slices,)
    """
    if slice_overlap_matrix.size == 0:
        return np.array([], dtype=np.float32)

    return (1.0 - np.prod(1.0 - slice_overlap_matrix, axis=0)).astype(np.float32)