import os
import pandas as pd
import numpy as np
from typing import List


# ---------------------------------------------------------------------
# Logistic confidence utilities
# ---------------------------------------------------------------------

def logistic_confidence(mean_z,
                        k=1.0,
                        x0=0.0,
                        lower=0.01,
                        upper=0.99):
    """
    Map a standardized feature value `mean_z` into a bounded confidence score
    using a logistic curve.

    Parameters
    ----------
    mean_z : float
        Standardized feature value.
    k : float
        Logistic slope.
    x0 : float
        Logistic midpoint.
    lower, upper : float
        Output confidence bounds.

    Returns
    -------
    float
        Confidence in [lower, upper].
    """
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
    """
    Compute annotation confidence from duration and bandwidth features
    using z-scored features and a logistic mapping.

    Parameters
    ----------
    duration : float
        Annotation duration in seconds.
    bandwidth : float
        Annotation bandwidth in Hz.
    duration_stats, bandwidth_stats : dict
        Dicts with keys {"mean", "std"} for z-scoring.
    combine : {'mean', (w1, w2)}
        How to combine duration and bandwidth z-scores.
    k, x0, lower, upper : float
        Logistic parameters.
    clip_z : float
        Clip z-scores to [-clip_z, clip_z].

    Returns
    -------
    float
        Confidence in [lower, upper].
    """
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


# ---------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------

def load_annotations(annotation_dir, audio_file):
    """
    Load Raven selection table annotations for a given audio file.

    Parameters
    ----------
    annotation_dir : str
        Directory containing annotation files.
    audio_file : str
        Audio filename whose base name determines the annotation file.

    Returns
    -------
    pandas.DataFrame
        Annotation table with standardized 'Annotation' text.
    """
    base = os.path.splitext(audio_file)[0]
    ann_filename = f"{base}.Table.1.selections.txt"
    ann_path = os.path.join(annotation_dir, ann_filename)

    if not os.path.exists(ann_path):
        return pd.DataFrame(columns=['Annotation', 'Begin Time (s)', 'End Time (s)'])

    df = pd.read_csv(ann_path, sep='\t')
    df['Annotation'] = df['Annotation'].astype(str).str.strip().str.lower()
    return df


# ---------------------------------------------------------------------
# Annotation confidence wrapper
# ---------------------------------------------------------------------

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

    Parameters
    ----------
    row : pandas.Series
        Annotation row with Begin/End times and optional Bandwidth.
    q2_confidence : float
        Confidence assigned to Q2 annotations when continuous mode is off.
    use_continuous_confidence : bool
        Whether to use logistic feature-based confidence.
    duration_stats, bandwidth_stats : dict
        Stats for z-scoring duration/bandwidth.
    logistic_params : dict
        Additional parameters for logistic mapping.

    Returns
    -------
    float
        Confidence in [0,1].
    """
    ann_text = str(row.get('Annotation', '')).lower()

    if not use_continuous_confidence:
        return q2_confidence if 'q2' in ann_text else 1.0

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


# ---------------------------------------------------------------------
# Slice-level confidences
# ---------------------------------------------------------------------

def slice_binary_confidences(
    annotations,
    clip_start,
    clip_end,
    n_slices=16,
    q2_confidence=0.75,
    use_continuous_confidence=False,
    duration_stats=None,
    bandwidth_stats=None,
    logistic_params=None,
):
    """
    Compute per-slice binary confidences by applying `has_frog_call`
    to each of the `n_slices` subwindows of a clip.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Annotation table.
    clip_start, clip_end : float
        Bounds of the 5s clip.
    n_slices : int
        Number of temporal slices (default 16).
    q2_confidence, use_continuous_confidence, duration_stats, bandwidth_stats, logistic_params
        Passed through to `compute_annotation_confidence`.

    Returns
    -------
    np.ndarray
        Array of shape (n_slices,) with slice-level confidences.
    """
    clip_start = float(clip_start)
    clip_end = float(clip_end)
    if clip_end <= clip_start or n_slices < 1:
        return np.zeros(int(n_slices), dtype=np.float32)

    total_dur = clip_end - clip_start
    slice_dur = total_dur / float(n_slices)

    out = np.zeros(int(n_slices), dtype=np.float32)

    for i in range(n_slices):
        s_start = clip_start + i * slice_dur
        s_end = s_start + slice_dur

        out[i] = has_frog_call(
            annotations=annotations,
            clip_start=s_start,
            clip_end=s_end,
            q2_confidence=q2_confidence,
            use_continuous_confidence=use_continuous_confidence,
            duration_stats=duration_stats,
            bandwidth_stats=bandwidth_stats,
            logistic_params=logistic_params,
        )

    return out


# ---------------------------------------------------------------------
# Clip-level confidence
# ---------------------------------------------------------------------

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
    Compute clip-level confidence that a frog call occurs within the window.

    The score is:
        max_over_annotations( overlap_fraction * annotation_confidence )

    where overlap_fraction = overlap_duration / annotation_duration.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Annotation table.
    clip_start, clip_end : float
        Window bounds.
    q2_confidence, use_continuous_confidence, duration_stats, bandwidth_stats, logistic_params
        Passed through to `compute_annotation_confidence`.

    Returns
    -------
    float
        Confidence in [0,1].
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
        overlap_end = min(clip_end, ann_end)
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


# ---------------------------------------------------------------------
# Count-distribution utilities
# ---------------------------------------------------------------------

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
    """
    Return a list of per-annotation weights for a clip, where each weight is:

        overlap_fraction * annotation_confidence

    This is used as input to `soft_count_distribution`.

    Parameters
    ----------
    annotations : pandas.DataFrame
    clip_start, clip_end : float
        Clip bounds.
    q2_confidence, use_continuous_confidence, duration_stats, bandwidth_stats, logistic_params
        Passed through to `compute_annotation_confidence`.

    Returns
    -------
    list of float
        Per-annotation weights in [0,1].
    """
    weights = []

    clip_start = float(clip_start)
    clip_end = float(clip_end)
    if clip_end <= clip_start:
        return weights

    for _, row in annotations.iterrows():
        ann_text = str(row.get('Annotation', '')).lower()
        if 'white dot' not in ann_text:
            continue

        try:
            ann_start = float(row['Begin Time (s)'])
            ann_end = float(row['End Time (s)'])
        except Exception:
            continue

        duration = ann_end - ann_start
        if duration <= 0:
            continue

        overlap_start = max(clip_start, ann_start)
        overlap_end = min(clip_end, ann_end)
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

        weight = float(np.clip(fraction * confidence, 0.0, 1.0))
        if weight > 0.0:
            weights.append(weight)

    return weights


def soft_count_distribution(weights: List[float], max_bin: int = 4) -> np.ndarray:
    """
    Compute a truncated distribution over {0, 1, ..., max_bin} calls
    given independent Bernoulli probabilities for each potential call.

    The final bin accumulates all probability mass for counts > max_bin.

    Parameters
    ----------
    weights : list of float
        Independent Bernoulli probabilities in [0,1].
    max_bin : int
        Maximum count bin.

    Returns
    -------
    np.ndarray
        Probability vector of length max_bin+1.
    """
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