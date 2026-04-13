import os
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================
# IMPORT YOUR CURRENT LOGIC (copied here verbatim)
# =====================================================

def has_frog_call_old(annotations, clip_start, clip_end):
    """
    OLD: max(overlap_fraction * confidence) across annotations,
    but confidence=1.0 always, no early skips, slightly different ordering.
    """
    max_score = 0.0

    for _, row in annotations.iterrows():
        if 'white dot' not in str(row['Annotation']).lower():
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
        score = fraction  # old code: confidence=1 always

        max_score = max(max_score, score)

    return max_score


def get_frog_call_weights_old(annotations, clip_start, clip_end):
    """
    OLD weight computation.
    - confidence: 0.75 if 'q2' else 1.0
    - No numeric clamping
    - No early clip_end <= clip_start check
    """
    weights = []

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
        confidence = 0.75 if 'q2' in ann_text else 1.0
        weight = fraction * confidence

        if weight > 0:
            weights.append(weight)

    return weights


def soft_count_distribution_old(weights, max_bin=4):
    """
    Original distribution logic (variable-length vector).
    """
    if not weights:
        dist = np.zeros(max_bin + 1, dtype=np.float32)
        dist[0] = 1.0
        return dist

    probs = np.array(weights, dtype=np.float32)
    dist = np.array([1.0], dtype=np.float32)

    for p in probs:
        left = dist * (1.0 - p)
        right = np.concatenate((np.zeros(1, dtype=np.float32), dist * p))
        dist = np.concatenate((left, np.zeros(1, dtype=np.float32))) + right

    # truncate / pad to max_bin
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


# =====================================================
# IMPORT YOUR CURRENT PRODUCTION LOGIC
# =====================================================

from frog_perch.utils.annotations import (
    load_annotations,
    has_frog_call,
    get_frog_call_weights,
    soft_count_distribution
)

# =====================================================
# COMPARISON SCRIPT
# =====================================================

def compare_annotation_logic(annotation_dir, audio_dir, clip_length=5.0, max_bin=4):
    annotation_dir = Path(annotation_dir)
    audio_dir = Path(audio_dir)

    audio_files = sorted([f for f in audio_dir.iterdir() if f.suffix in [".wav", ".flac", ".mp3"]])

    changed = []

    for audio_file in audio_files:
        ann = load_annotations(annotation_dir, audio_file.name)

        clip_start = 0.0
        clip_end = clip_length

        # OLD LOGIC
        old_bin = has_frog_call_old(ann, clip_start, clip_end)
        old_weights = get_frog_call_weights_old(ann, clip_start, clip_end)
        old_dist = soft_count_distribution_old(old_weights, max_bin=max_bin)

        # NEW LOGIC
        new_bin = has_frog_call(ann, clip_start, clip_end)
        new_weights = get_frog_call_weights(ann, clip_start, clip_end)
        new_dist = soft_count_distribution(new_weights, max_bin=max_bin)

        # Compare
        if (
            not np.isclose(old_bin, new_bin)
            or not np.allclose(old_dist, new_dist)
            or old_weights != new_weights
        ):
            changed.append({
                "audio": audio_file.name,
                "old_bin": old_bin,
                "new_bin": new_bin,
                "old_weights": old_weights,
                "new_weights": new_weights,
                "old_dist": old_dist,
                "new_dist": new_dist,
            })

    # Summary
    print(f"\nCompared {len(audio_files)} audio files.")
    print(f"{len(changed)} files differ between old and new logic.\n")

    for item in changed:
        print("--------------------------------------------------------")
        print(f"AUDIO: {item['audio']}")
        print(f"  old_binary = {item['old_bin']:.4f}")
        print(f"  new_binary = {item['new_bin']:.4f}")
        print(f"  old_weights = {item['old_weights']}")
        print(f"  new_weights = {item['new_weights']}")
        print(f"  old_dist = {item['old_dist']}")
        print(f"  new_dist = {item['new_dist']}")
        print()

    return changed


# =====================================================
# USAGE
# =====================================================
if __name__ == "__main__":
    annotation_dir = "/home/breallis/datasets/frog_calls/round_2/"
    audio_dir = "/home/breallis/datasets/frog_calls/round_2/"

    compare_annotation_logic(annotation_dir, audio_dir)
