# utils/annotations.py
import os
import pandas as pd
import numpy as np

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

def has_frog_call(annotations, clip_start, clip_end):
    """
    Port of _has_frog_call in your PyTorch dataset. Returns fractional overlap * confidence.
    """
    max_overlap = 0.0
    confidence = 1.0
    for _, row in annotations.iterrows():
        if 'white dot' in row['Annotation']:
            ann_start = float(row['Begin Time (s)'])
            ann_end = float(row['End Time (s)'])
            overlap_start = max(clip_start, ann_start)
            overlap_end = min(clip_end, ann_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if ann_end - ann_start <= 0:
                continue
            fraction = overlap / (ann_end - ann_start)
            if fraction > max_overlap:
                max_overlap = fraction
                confidence = 0.75 if 'q2' in row['Annotation'] else 1.0
    return max_overlap * confidence

def get_frog_call_weights(annotations, clip_start, clip_end):
    """
    Port of get_frog_call_weights. Returns list of weights in [0,1].
    """
    weights = []
    for _, row in annotations.iterrows():
        if 'white dot' in row['Annotation']:
            ann_start = float(row['Begin Time (s)'])
            ann_end = float(row['End Time (s)'])
            overlap_start = max(clip_start, ann_start)
            overlap_end = min(clip_end, ann_end)
            overlap = max(0.0, overlap_end - overlap_start)
            if ann_end - ann_start <= 0:
                continue
            fraction = overlap / (ann_end - ann_start)
            confidence = 0.75 if 'q2' in row['Annotation'] else 1.0
            weight = fraction * confidence
            if weight > 0:
                weights.append(weight)
    return weights

def soft_count_distribution(weights, max_bin=4):
    """
    Port of soft_count_distribution from torch to numpy.
    Returns numpy array shape (max_bin+1,) summing to 1.0
    """
    if not weights:
        dist = np.zeros(max_bin + 1, dtype=np.float32)
        dist[0] = 1.0
        return dist

    probs = np.array(weights, dtype=np.float32)
    dist = np.array([1.0], dtype=np.float32)  # initial zero-count prob

    for p in probs:
        left = dist * (1.0 - p)
        right = np.concatenate((np.zeros(1, dtype=np.float32), dist * p))
        dist = np.concatenate((left, np.zeros(1, dtype=np.float32))) + right
    # pad/truncate
    if len(dist) <= max_bin:
        padded = np.zeros(max_bin + 1, dtype=np.float32)
        padded[:len(dist)] = dist
        dist = padded
    else:
        tail = dist[max_bin:].sum()
        dist = dist[:max_bin+1]
        dist[max_bin] += tail
    s = dist.sum()
    if s > 0:
        dist = dist / s
    return dist.astype(np.float32)
