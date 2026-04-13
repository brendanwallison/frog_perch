#!/usr/bin/env python3
"""
infer_directory.py

Runs inference on all audio files in INPUT_DIR and outputs one CSV per file.

Defaults come from infer_config.py.
Argparse overrides are optional.
"""

import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf

# Make project importable
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# project imports
from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.models.perch_wrapper import PerchWrapper
import frog_perch.config as config

# local config with defaults
import experiments.infer_config as cfg


# -------------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------------

def list_audio_files(root: Path, exts):
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files


def windows_for_file(total_seconds, clip_seconds, stride):
    max_start = max(0.0, total_seconds - clip_seconds)
    if max_start <= 0:
        return np.array([], dtype=float)
    return np.arange(0.0, max_start, stride)


def load_model_for_count(ckpt_path: str):
    print(f"[INFO] Loading model: {ckpt_path}")
    model = tf.keras.models.load_model(ckpt_path, compile=False)
    try:
        logits_layer = model.get_layer("count_logits")
    except ValueError:
        raise RuntimeError("Model missing 'count_logits' layer")
    model_for_pred = tf.keras.Model(
        inputs=model.inputs,
        outputs=[logits_layer.output, model.output],  # logits, probs
    )
    return model_for_pred


# -------------------------------------------------------------------------------------
# Aggregation helpers
# -------------------------------------------------------------------------------------

def compute_derived_counts(probs_row: np.ndarray):
    """
    probs_row: 1D array shape (K,) with probabilities for k=0..K-1
    Returns (p_present, expected_count, factored_expected_count)
    """
    k_vals = np.arange(probs_row.shape[-1], dtype=np.float32)
    expected = float(np.sum(probs_row * k_vals))
    p0 = float(probs_row[0])
    p_present = 1.0 - p0
    if p_present > 0.0:
        factored = expected / p_present
    else:
        factored = 0.0
    return p_present, expected, factored


# -------------------------------------------------------------------------------------
# Main inference for a single file
# -------------------------------------------------------------------------------------

def process_file(path_audio, model, perch, out_csv, batch_size, stride):
    sample_rate = config.DATASET_SAMPLE_RATE
    clip_seconds = config.CLIP_DURATION_SECONDS
    clip_samples = int(sample_rate * clip_seconds)
    perch_sr = config.PERCH_SAMPLE_RATE
    perch_samples = config.PERCH_CLIP_SAMPLES

    print(f"[INFO] Processing {path_audio.name}")
    audio, sr = load_audio(str(path_audio), target_sr=sample_rate)
    total_seconds = len(audio) / sample_rate

    starts = windows_for_file(total_seconds, clip_seconds, stride)

    # empty CSV case
    header = [
        "window_idx", "start_sec", "end_sec",
        "p0", "p1", "p2", "p3", "p4",
        "p_present", "expected_count", "factored_expected_count"
    ]
    if len(starts) == 0:
        print(f"[INFO] No windows (file < clip length): {path_audio.name}")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
        return

    rows = []
    emb_batch = []
    meta_batch = []

    for idx, st in enumerate(starts):
        st_samp = int(round(st * sample_rate))
        ed_samp = st_samp + clip_samples

        # deterministic slice/pad
        if ed_samp <= len(audio):
            clip = audio[st_samp:ed_samp]
        else:
            clip = np.zeros(clip_samples, dtype=np.float32)
            val = max(0, len(audio) - st_samp)
            if val > 0:
                clip[:val] = audio[st_samp:st_samp+val]

        # resample to Perch
        clip_p = resample_array(clip, sample_rate, perch_sr)
        if len(clip_p) < perch_samples:
            clip_p = np.pad(clip_p, (0, perch_samples - len(clip_p)))
        clip_p = clip_p[:perch_samples]

        embedding = perch.get_spatial_embedding(clip_p).astype(np.float32)

        emb_batch.append(embedding)
        meta_batch.append((idx, float(st), float(st + clip_seconds)))

        if len(emb_batch) >= batch_size:
            logits, probs = model.predict(np.stack(emb_batch), verbose=0)
            probs = np.asarray(probs)
            for (i, s, e), p in zip(meta_batch, probs):
                p_present, expected, factored = compute_derived_counts(p)
                rows.append((i, s, e, p, p_present, expected, factored))
            emb_batch, meta_batch = [], []

    # flush last batch
    if emb_batch:
        logits, probs = model.predict(np.stack(emb_batch), verbose=0)
        probs = np.asarray(probs)
        for (i, s, e), p in zip(meta_batch, probs):
            p_present, expected, factored = compute_derived_counts(p)
            rows.append((i, s, e, p, p_present, expected, factored))

    # write CSV
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i, s, e, p, p_present, expected, factored in rows:
            w.writerow([
                int(i), float(s), float(e),
                float(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                float(p_present), float(expected), float(factored)
            ])

    print(f"[INFO] Wrote {len(rows)} rows → {out_csv}")


# -------------------------------------------------------------------------------------
# Wrapper main
# -------------------------------------------------------------------------------------

def main():
    # argparse ONLY for optional overrides
    P = argparse.ArgumentParser(add_help=False)
    P.add_argument("--ckpt")
    P.add_argument("--input_dir")
    P.add_argument("--output_dir")
    P.add_argument("--batch_size", type=int)
    P.add_argument("--stride", type=float)
    P.add_argument("--ext")
    P.add_argument("--overwrite", action="store_true")
    args = P.parse_args()

    # fallback to config values
    ckpt = args.ckpt or cfg.CKPT
    input_dir = Path(args.input_dir or cfg.INPUT_DIR)
    output_dir = Path(args.output_dir or cfg.OUTPUT_DIR)
    batch_size = args.batch_size or cfg.BATCH_SIZE
    stride = args.stride or cfg.STRIDE
    exts = (args.ext.split(",") if args.ext else cfg.EXT)
    overwrite = args.overwrite or cfg.OVERWRITE

    output_dir.mkdir(parents=True, exist_ok=True)

    # load model + perch
    model = load_model_for_count(ckpt)
    perch = PerchWrapper()

    test = perch.get_spatial_embedding(np.random.uniform(-1, 1, 5 * 32000).astype(np.float32))

    B = 8  # batch size you want to test
    batch = np.random.uniform(-1, 1, (B, 5 * 32000)).astype(np.float32)
    embeddings = perch.get_spatial_embedding(batch)
    # embeddings.shape -> (B, ...)  # one embedding per input
    
    audio_files = list_audio_files(input_dir, exts)
    if not audio_files:
        print(f"[WARN] No audio files in {input_dir}")
        return

    for af in audio_files:
        out_csv = output_dir / (af.stem + ".csv")
        if out_csv.exists() and not overwrite:
            print(f"[SKIP] {af.name} (exists)")
            continue
        try:
            process_file(af, model, perch, out_csv, batch_size, stride)
        except Exception as e:
            print(f"[ERROR] Failed processing {af}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
