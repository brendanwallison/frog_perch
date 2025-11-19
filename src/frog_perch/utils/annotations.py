import os
import pandas as pd
import numpy as np
from typing import List

def load_annotations(annotation_dir, audio_file):
    """
    Return DataFrame with columns ['Annotation', 'Begin Time (s)', 'End Time (s)'].
    If annotation file missing, return empty DataFrame (meaning no calls).
    """
    base = os.path.splitext(audio_file)[0]
    ann_filename = f"{base}.Table.1.selections.txt"
    ann_path = os.path.join(annotation_dir, ann_filename)
    if not os.path.exists(ann_path):
        return pd.DataFrame(columns=['Annotation', 'Begin Time (s)', 'End Time (s)'])
    df = pd.read_csv(ann_path, sep='\t')
    df['Annotation'] = df['Annotation'].astype(str).str.strip().str.lower()
    return df


def has_frog_call(annotations, clip_start, clip_end, q2_confidence: float = 0.75):
    """
    Return max(overlap_fraction * confidence) across annotations.

    q2_confidence: value to use when annotation text contains 'q2'.
                   Default maintains original behavior (0.75).
    """
    max_score = 0.0

    for _, row in annotations.iterrows():
        if 'white dot' not in row['Annotation']:
            continue

        ann_start = float(row['Begin Time (s)'])
        ann_end = float(row['End Time (s)'])
        duration = ann_end - ann_start
        if duration <= 0:
            continue

        # Compute overlap between annotation and clip
        overlap_start = max(clip_start, ann_start)
        overlap_end = min(clip_end, ann_end)
        overlap = overlap_end - overlap_start
        if overlap <= 0:
            continue

        fraction = overlap / duration

        # Quality-dependent confidence
        confidence = q2_confidence if 'q2' in row['Annotation'] else 1.0

        score = fraction * confidence
        max_score = max(max_score, score)

    return max_score


def get_frog_call_weights(
    annotations,
    clip_start: float,
    clip_end: float,
    q2_confidence: float = 0.75
) -> List[float]:
    """
    Compute list of weights in [0,1] for overlapping annotations.

    q2_confidence: weight applied to 'q2' annotations.
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

        # Quality-dependent confidence
        confidence = q2_confidence if 'q2' in ann_text else 1.0

        weight = fraction * confidence
        weight = max(0.0, min(1.0, float(weight)))

        if weight > 0.0:
            weights.append(weight)

    return weights


def soft_count_distribution(weights: List[float], max_bin: int = 4) -> np.ndarray:
    """
    Compute soft distribution over counts (0..max_bin).
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
        p = float(max(0.0, min(1.0, p)))
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
        dist = dist / s
    else:
        dist[:] = 0
        dist[0] = 1.0

    return dist.astype(np.float32)
