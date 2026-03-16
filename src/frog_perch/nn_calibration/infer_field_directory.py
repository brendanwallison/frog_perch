#!/usr/bin/env python3
"""
infer_field_directory.py

Bulk inference script for processing field audio. 
Extracts Perch embeddings, runs the custom Keras model to get expected count (mu) 
and variance (var), and calculates the necessary environmental noise covariates.

Outputs a CSV per audio file that is ready for Bayesian calibration.
"""
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import frog_perch.config as config
from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.models.perch_wrapper import PerchWrapper

# Import from our shared core module
from frog_perch.nn_calibration.feature_extraction import (
    calculate_bandpass_features,
    load_custom_model,
    ensure_1d_probs,
    calculate_window_moments
)

def list_audio_files(root: Path, exts):
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files

def process_file(path_audio, model, perch, out_csv, batch_size, step_seconds):
    sample_rate = config.DATASET_SAMPLE_RATE
    clip_seconds = config.CLIP_DURATION_SECONDS
    clip_samples = int(sample_rate * clip_seconds)
    perch_sr = config.PERCH_SAMPLE_RATE
    perch_samples = config.PERCH_CLIP_SAMPLES

    print(f"[INFO] Processing {path_audio.name} (step={step_seconds}s)")
    audio, sr = load_audio(str(path_audio), target_sr=sample_rate)
    total_seconds = len(audio) / sample_rate

    # Calculate start times for windows
    max_start = max(0.0, total_seconds - clip_seconds)
    if max_start < 0:
        print(f"[INFO] File shorter than clip length: {path_audio.name}")
        pd.DataFrame(columns=["time_sec", "nn_mu", "nn_var", "log_mean_rms_1000_1500"]).to_csv(out_csv, index=False)
        return

    starts = np.arange(0.0, max_start + 1e-8, step_seconds)
    
    rows = []
    emb_batch = []
    meta_batch = []
    
    def _flush_batch(embs, metas):
        """Helper to process a batch of embeddings and append to rows."""
        X = np.stack(embs)
        logits = model.predict(X, verbose=0)
        probs_batch = ensure_1d_probs(logits)
        
        # calculate_window_moments expects shape [B, T], returns (mu_array, var_array)
        batch_mu, batch_var = calculate_window_moments(probs_batch)
        
        for idx, (st, band_feats) in enumerate(metas):
            res = {
                "time_sec": st,
                "nn_mu": batch_mu[idx],
                "nn_var": batch_var[idx]
            }
            res.update(band_feats)
            rows.append(res)

    for st in starts:
        st_samp = int(round(st * sample_rate))
        ed_samp = st_samp + clip_samples

        # 1. Audio Slicing
        if ed_samp <= len(audio):
            clip = audio[st_samp:ed_samp]
        else:
            clip = np.zeros(clip_samples, dtype=np.float32)
            val = max(0, len(audio) - st_samp)
            if val > 0:
                clip[:val] = audio[st_samp:st_samp + val]

        # 2. Extract Acoustic Covariates (Bandpass RMS) using shared function
        band_feats = calculate_bandpass_features(clip, sample_rate)

        # 3. Extract Perch Embeddings
        clip_p = resample_array(clip, sample_rate, perch_sr)
        if len(clip_p) < perch_samples:
            clip_p = np.pad(clip_p, (0, perch_samples - len(clip_p)))
        clip_p = clip_p[:perch_samples]

        embedding = perch.get_spatial_embedding(clip_p).astype(np.float32)

        emb_batch.append(embedding)
        meta_batch.append((float(st), band_feats))

        # 4. Process Batch if full
        if len(emb_batch) >= batch_size:
            _flush_batch(emb_batch, meta_batch)
            emb_batch, meta_batch = [], []

    # Flush remaining
    if emb_batch:
        _flush_batch(emb_batch, meta_batch)

    # Save to CSV
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote {len(df_out)} windows -> {out_csv}")

def main():
    # Setup default paths based on the local environment
    default_ckpt = os.path.join(config.CHECKPOINT_DIR, "pool=slice_loss=slice_x0=-3.0_k=1.0.keras")
    default_input = "/home/breallis/datasets/frog_calls/gabon_full/P2"
    default_output = "/home/breallis/datasets/frog_calls/gabon_full/P2_nn_features"

    parser = argparse.ArgumentParser(description="Extract NN counts and noise covariates from field audio.")
    parser.add_argument("--ckpt", default=default_ckpt, help="Path to trained Keras model")
    parser.add_argument("--input_dir", default=default_input, help="Directory containing raw field audio.")
    parser.add_argument("--output_dir", default=default_output, help="Directory to save output CSVs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--step_seconds", type=float, default=5.0, help="Window stride. Default is 5.0 (non-overlapping).")
    parser.add_argument("--ext", default="wav,flac,mp3")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = args.ext.split(",")

    print(f"[INFO] Loading Model from {args.ckpt}...")
    model = load_custom_model(args.ckpt)
    
    print("[INFO] Loading Perch Wrapper...")
    perch = PerchWrapper()
    _ = perch.get_spatial_embedding(np.zeros(5 * 32000, dtype=np.float32)) # Warmup

    audio_files = list_audio_files(input_dir, exts)
    if not audio_files:
        print(f"[WARN] No audio files found in {input_dir}")
        return

    for af in audio_files:
        out_csv = output_dir / (af.stem + ".csv")
        if out_csv.exists() and not args.overwrite:
            print(f"[SKIP] {af.name} (exists)")
            continue
        try:
            process_file(af, model, perch, out_csv, args.batch_size, args.step_seconds)
        except Exception as e:
            print(f"[ERROR] Failed processing {af}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()