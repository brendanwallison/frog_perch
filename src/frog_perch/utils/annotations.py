import os
import pandas as pd
import numpy as np
from typing import List

# ================================================================
#  Robust z-score + logistic confidence (unchanged)
# ================================================================

# def robust_zscore(x, median=None, mad=None, clip=5.0):
#     x = np.asarray(x, dtype=np.float32)
#     if median is None:
#         median = np.median(x)
#     if mad is None:
#         mad = np.median(np.abs(x - median))
#     sigma = max(1e-6, 1.4826 * mad)
#     z = (x - median) / sigma
#     if clip is not None:
#         z = np.clip(z, -clip, clip)
#     return z, median, mad


def logistic_confidence(mean_z,
                        k=1.0,
                        x0=0.0,
                        lower=0.01,
                        upper=0.99):
    sigmoid = 1.0 / (1.0 + np.exp(-k * (mean_z - x0)))
    conf = lower + (upper - lower) * sigmoid
    return float(np.clip(conf, lower, upper))


def annotation_confidence_from_features(duration,
                                        bandwidth,
                                        duration_stats=None,
                                        bandwidth_stats=None,
                                        combine='mean',
                                        k=1.0, x0=0.0,
                                        lower=0.01, upper=0.99,
                                        clip_z=5.0):
    if duration_stats is None or bandwidth_stats is None:
        raise ValueError("Pass precomputed duration_stats and bandwidth_stats (mean, std).")

    d_mean = duration_stats["mean"]
    d_std = duration_stats["std"]
    b_mean = bandwidth_stats["mean"]
    b_std = bandwidth_stats["std"]

    z_d = (float(duration) - d_mean) / max(1e-6, d_std)
    z_b = (float(bandwidth) - b_mean) / max(1e-6, b_std)

    z_d = float(np.clip(z_d, -clip_z, clip_z))
    z_b = float(np.clip(z_b, -clip_z, clip_z))

    if isinstance(combine, (list, tuple)) and len(combine) == 2:
        w1, w2 = combine
        mean_z = w1 * z_d + w2 * z_b
    else:
        mean_z = 0.5 * (z_d + z_b)

    return logistic_confidence(mean_z, k=k, x0=x0, lower=lower, upper=upper)


# ================================================================
#  Annotation loading
# ================================================================

def load_annotations(annotation_dir, audio_file):
    base = os.path.splitext(audio_file)[0]
    ann_filename = f"{base}.Table.1.selections.txt"
    ann_path = os.path.join(annotation_dir, ann_filename)
    if not os.path.exists(ann_path):
        return pd.DataFrame(columns=['Annotation', 'Begin Time (s)', 'End Time (s)'])
    df = pd.read_csv(ann_path, sep='\t')
    df['Annotation'] = df['Annotation'].astype(str).str.strip().str.lower()
    return df


# ================================================================
#  NEW: Shared confidence handler
# ================================================================

def compute_annotation_confidence(
    row,
    q2_confidence: float = 0.75,
    use_continuous_confidence: bool = False,
    duration_stats=None,
    bandwidth_stats=None,
    logistic_params: dict = None,
):
    """
    Compute confidence for a single annotation row.

    If use_continuous_confidence=False:
        Uses Q2 hardcoded penalty (original behavior).
    If True:
        Uses logistic function with duration/bandwidth features.
    """
    ann_text = str(row.get('Annotation', '')).lower()

    if not use_continuous_confidence:
        # original "q2 or full confidence"
        return q2_confidence if 'q2' in ann_text else 1.0

    # --- continuous mode ---
    logistic_params = logistic_params or {}

    duration = float(row.get('End Time (s)', 0.0)) - float(row.get('Begin Time (s)', 0.0))
    try:
        bandwidth = float(row.get('Bandwidth (Hz)', 0.0))
    except Exception:
        bandwidth = 0.0

    return annotation_confidence_from_features(
        duration=duration,
        bandwidth=bandwidth,
        duration_stats=duration_stats,
        bandwidth_stats=bandwidth_stats,
        **logistic_params
    )


# ================================================================
#  has_frog_call (DRY)
# ================================================================

def has_frog_call(
    annotations,
    clip_start,
    clip_end,
    q2_confidence: float = 0.75,
    use_continuous_confidence: bool = False,
    duration_stats=None,
    bandwidth_stats=None,
    logistic_params: dict = None,
):
    """
    Return max(overlap_fraction * confidence) over annotations.
    """
    max_score = 0.0

    for _, row in annotations.iterrows():
        if 'white' not in str(row.get('Annotation', '')).lower():
            continue

        ann_start = float(row['Begin Time (s)'])
        ann_end = float(row['End Time (s)'])
        duration = ann_end - ann_start
        if duration <= 0:
            continue

        overlap_start = max(clip_start, ann_start)
        overlap_end   = min(clip_end, ann_end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue

        fraction = overlap / duration

        confidence = compute_annotation_confidence(
            row=row,
            q2_confidence=q2_confidence,
            use_continuous_confidence=use_continuous_confidence,
            duration_stats=duration_stats,
            bandwidth_stats=bandwidth_stats,
            logistic_params=logistic_params,
        )

        score = fraction * confidence
        max_score = max(max_score, score)

    return max_score


# ================================================================
#  get_frog_call_weights (DRY)
# ================================================================

def get_frog_call_weights(
    annotations,
    clip_start: float,
    clip_end: float,
    q2_confidence: float = 0.75,
    use_continuous_confidence: bool = False,
    duration_stats=None,
    bandwidth_stats=None,
    logistic_params: dict = None,
) -> List[float]:
    weights = []

    clip_start = float(clip_start)
    clip_end   = float(clip_end)
    if clip_end <= clip_start:
        return weights

    for _, row in annotations.iterrows():
        ann_text = str(row.get('Annotation', '')).lower()
        if 'white dot' not in ann_text:
            continue

        try:
            ann_start = float(row['Begin Time (s)'])
            ann_end   = float(row['End Time (s)'])
        except Exception:
            continue

        duration = ann_end - ann_start
        if duration <= 0:
            continue

        overlap_start = max(clip_start, ann_start)
        overlap_end   = min(clip_end, ann_end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue

        fraction = overlap / duration

        confidence = compute_annotation_confidence(
            row=row,
            q2_confidence=q2_confidence,
            use_continuous_confidence=use_continuous_confidence,
            duration_stats=duration_stats,
            bandwidth_stats=bandwidth_stats,
            logistic_params=logistic_params,
        )

        weight = fraction * confidence
        weight = float(np.clip(weight, 0.0, 1.0))

        if weight > 0.0:
            weights.append(weight)

    return weights


# ================================================================
#  soft_count_distribution (unchanged)
# ================================================================

def soft_count_distribution(weights: List[float], max_bin: int = 4) -> np.ndarray:
    max_bin = int(max_bin)
    assert max_bin >= 0

    if not weights:
        dist = np.zeros(max_bin + 1, dtype=np.float32)
        dist[0] = 1.0
        return dist

    probs = np.array(weights, dtype=np.float32)
    dist = np.zeros(max_bin + 1, dtype=np.float32)
    dist[0] = 1.0

    for p in probs:
        p = float(np.clip(p, 0.0, 1.0))
        if p == 0.0:
            continue
        if p == 1.0:
            new = np.zeros_like(dist)
            new[1:] = dist[:-1]
            new[-1] += dist[-1]
            dist = new
            continue

        new = np.zeros_like(dist, dtype=np.float32)
        new += dist * (1.0 - p)
        new[1:] += dist[:-1] * p
        new[-1] += dist[-1] * p
        dist = new

    s = float(dist.sum())
    if s > 0:
        dist /= s
    else:
        dist[:] = 0
        dist[0] = 1.0

    return dist.astype(np.float32)
