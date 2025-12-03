#!/usr/bin/env python3
"""
collect_errors.py

Usage:
    python collect_errors.py --ckpt /path/to/model.keras \
        --n 50 --output_dir ./diag \
        [--audio_dir /path/to/audio] [--annotation_dir /path/to/ann]
"""

import os
import argparse
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import tensorflow as tf

from frog_perch.datasets.frog_dataset import FrogPerchDataset

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def safe_makedirs(p):
    os.makedirs(p, exist_ok=True)


def save_clip_wav(audio_root, audio_file, start_sample, clip_samples, sr, out_path):
    """Extract a clip from a WAV file and save it."""
    audio_path = os.path.join(audio_root, audio_file)
    data, file_sr = sf.read(audio_path, dtype='float32')

    # We assume the saved WAV files already match val_obj.sample_rate
    # If not, resampling should happen here, but FrogPerchDataset normally
    # provides properly-resampled audio.
    if file_sr != sr:
        pass

    end = start_sample + clip_samples
    start = max(0, start_sample)

    if end <= len(data):
        segment = data[start:end]
    else:
        segment = np.zeros((clip_samples,), dtype=np.float32)
        valid = max(0, len(data) - start)
        if valid > 0:
            segment[:valid] = data[start:start+valid]

    sf.write(out_path, segment, sr)


# ---------------------------------------------------------------------
# Model loading (cleaned, no fallback)
# ---------------------------------------------------------------------

def load_full_model(ckpt_path):
    """
    Load a full Keras model saved with model.save().
    This replaces the previous try_load_model_or_weights().
    """
    try:
        print(f"[INFO] Loading full model from: {ckpt_path}")
        model = tf.keras.models.load_model(ckpt_path, compile=False)
        print("[INFO] Model loaded successfully.")
        return model
    except Exception as e:
        raise RuntimeError(
            f"Failed to load the model with tf.keras.models.load_model(). "
            f"Keras saved models must be loaded this way. "
            f"Original error:\n{e}"
        )


# ---------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------

def collect_errors(ckpt_path, label_mode, n, output_dir,
                   audio_dir=None, annotation_dir=None,
                   val_stride_sec=5.0, equalize_q2_val=True):

    # ------------------------------
    # 1. Prepare output dirs
    # ------------------------------
    out = Path(output_dir)
    fp_dir = out / "fp"
    fn_dir = out / "fn"
    safe_makedirs(fp_dir)
    safe_makedirs(fn_dir)

    meta_fp_path = out / "false_positives.jsonl"
    meta_fn_path = out / "false_negatives.jsonl"

    # ------------------------------
    # 2. Build validation dataset object
    # ------------------------------
    ds_args = {}
    if audio_dir:
        ds_args['audio_dir'] = audio_dir
    if annotation_dir:
        ds_args['annotation_dir'] = annotation_dir

    val_obj = FrogPerchDataset(
        train=False,
        val_stride_sec=val_stride_sec,
        label_mode=label_mode,
        equalize_q2_val=equalize_q2_val,
        **ds_args
    )

    clip_samples = val_obj.clip_samples
    sr = val_obj.sample_rate
    total_windows = len(val_obj)

    print(f"[INFO] Validation windows: {total_windows}")
    print(f"[INFO] clip_samples={clip_samples}, sample_rate={sr}")

    # ------------------------------
    # 3. Load model cleanly
    # ------------------------------
    model = load_full_model(ckpt_path)

    # Build a “model_for_pred” that returns logits when label_mode == 'count'
    if label_mode == 'count':
        try:
            logits_layer = model.get_layer('count_logits')
        except ValueError:
            raise RuntimeError(
                "Model does not contain a layer named 'count_logits'. "
                "Count models MUST include this layer for raw logits."
            )

        logits_output = logits_layer.output
        probs_output = model.output   # already softmax
        model_for_pred = tf.keras.Model(
            inputs=model.inputs,
            outputs=[logits_output, probs_output]
        )
    else:
        model_for_pred = model  # binary case

    # ------------------------------
    # 4. FP/FN collection
    # ------------------------------
    f_fp = open(meta_fp_path, "w")
    f_fn = open(meta_fn_path, "w")

    collected_fp = 0
    collected_fn = 0

    # bins for expected value in count mode
    if label_mode == 'count':
        _, first_label, _, _ = val_obj[0]
        K = len(first_label)
        bins = np.arange(K).astype(np.float32)

    for idx in range(total_windows):
        spatial_emb, label, audio_file, start_sample = val_obj[idx]

        inp = np.expand_dims(spatial_emb.astype(np.float32), 0)

        if label_mode == 'count':
            logits_np, probs_np = model_for_pred.predict(inp, verbose=0)
            logits_np = logits_np[0]
            probs_np = probs_np[0]

            exp_pred = float(np.sum(probs_np * bins))
            pred_bin = exp_pred >= 0.5

            true_bin = int(np.argmax(label)) > 0

            meta = {
                "audio_file": audio_file,
                "start_sample": int(start_sample),
                "start_sec": float(start_sample) / sr,
                "true_label": label.tolist(),
                "true_bin": bool(true_bin),
                "raw_logits": logits_np.tolist(),
                "probs": probs_np.tolist(),
                "expected_count": exp_pred,
                "pred_bin": bool(pred_bin),
            }

        else:
            # Binary
            logit_val = float(model_for_pred.predict(inp, verbose=0).squeeze())
            prob = 1 / (1 + np.exp(-logit_val))
            pred_bin = prob >= 0.5

            true_bin = bool(label > 0)

            meta = {
                "audio_file": audio_file,
                "start_sample": int(start_sample),
                "start_sec": float(start_sample) / sr,
                "true_label": float(label),
                "true_bin": bool(true_bin),
                "raw_logits": logit_val,
                "prob": prob,
                "pred_bin": bool(pred_bin),
            }

        # FP/FN logic
        if pred_bin and not true_bin:
            if collected_fp < n:
                fname = f"{idx:06d}__{audio_file}__{start_sample}.wav"
                save_clip_wav(val_obj.audio_dir, audio_file, start_sample, clip_samples, sr, fp_dir / fname)
                f_fp.write(json.dumps(meta) + "\n")
                f_fp.flush()
                collected_fp += 1
                print(f"[FP] idx={idx} ({collected_fp}/{n})")

        elif not pred_bin and true_bin:
            if collected_fn < n:
                fname = f"{idx:06d}__{audio_file}__{start_sample}.wav"
                save_clip_wav(val_obj.audio_dir, audio_file, start_sample, clip_samples, sr, fn_dir / fname)
                f_fn.write(json.dumps(meta) + "\n")
                f_fn.flush()
                collected_fn += 1
                print(f"[FN] idx={idx} ({collected_fn}/{n})")

        if collected_fp >= n and collected_fn >= n:
            print("[INFO] Finished collecting requested FP/FN.")
            break

    f_fp.close()
    f_fn.close()

    print(f"[DONE] FP={collected_fp}, FN={collected_fn}")
    print(f"[OUTPUT] FP clips written to: {fp_dir}")
    print(f"[OUTPUT] FN clips written to: {fn_dir}")
    print(f"[OUTPUT] Metadata files:\n  {meta_fp_path}\n  {meta_fn_path}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():

    ckpt = "/home/breallis/dev/frog_perch/checkpoints/pool=mlp_flat_loss=count_x0=-4.0_k=1.0.keras"
    label_mode = "count"
    n = 10
    output_dir = "misclassification_samples"
    audio_dir = '/home/breallis/datasets/frog_calls/round_2'     
    annotation_dir = '/home/breallis/datasets/frog_calls/round_2'     
    val_stride_sec = 5.0

    collect_errors(
        ckpt_path=ckpt,
        label_mode=label_mode,
        n=n,
        output_dir=output_dir,
        audio_dir=audio_dir,
        annotation_dir=annotation_dir,
        val_stride_sec=val_stride_sec
    )


if __name__ == "__main__":
    main()
