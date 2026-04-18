"""
feature_extraction_pipeline.py 

Functions for bulk processing field audio. Extracts Perch embeddings, 
runs the custom Keras model to get expected count (mu) and variance (var), 
and calculates necessary environmental noise covariates.

Outputs a CSV per audio file that is ready for Bayesian calibration.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path

from frog_perch.utils.audio import load_audio, resample_array
from frog_perch.nn_models.perch_wrapper import PerchWrapper

# Import from our shared core module
from frog_perch.nn_calibration.feature_extraction import (
    calculate_bandpass_features,
    load_custom_model,
    calculate_window_moments  # Removed ensure_1d_probs
)


def list_audio_files(root: Path, exts: list[str]) -> list[Path]:
    """Returns a sorted list of audio files in the given directory matching the extensions."""
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files


def process_file(
    path_audio: Path, 
    model, 
    perch: PerchWrapper, 
    out_csv: Path, 
    batch_size: int, 
    step_seconds: float, 
    config_dict: dict
) -> None:
    """Processes a single audio file, extracting features and model predictions."""
    
    sample_rate = config_dict.get("DATASET_SAMPLE_RATE", 32000)
    clip_seconds = config_dict.get("CLIP_DURATION_SECONDS", 5.0)
    clip_samples = int(sample_rate * clip_seconds)
    perch_sr = config_dict.get("PERCH_SAMPLE_RATE", 32000)
    perch_samples = config_dict.get("PERCH_CLIP_SAMPLES", 160000)

    print(f"[INFO] Processing {path_audio.name} (step={step_seconds}s)")
    audio, sr = load_audio(str(path_audio), target_sr=sample_rate)
    total_seconds = len(audio) / sample_rate

    # Calculate start times for windows
    max_start = max(0.0, total_seconds - clip_seconds)
    if max_start < 0:
        print(f"[INFO] File shorter than clip length: {path_audio.name}")
        pd.DataFrame(
            columns=["time_sec", "nn_mu", "nn_var", "log_mean_rms_1000_1500"]
        ).to_csv(out_csv, index=False)
        return

    starts = np.arange(0.0, max_start + 1e-8, step_seconds)
    
    rows = []
    emb_batch = []
    meta_batch = []
    
    def _flush_batch(embs, metas):
        """Helper to process a batch of embeddings and append to rows."""
        X = np.stack(embs)
        
        # UPDATED: model.predict now returns a dictionary of outputs
        preds_dict = model.predict(X, verbose=0)
        
        # UPDATED: calculate_window_moments extracts 'count_probs' directly from the dict
        batch_mu, batch_var = calculate_window_moments(preds_dict)
        
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


def process_directory(
    config_dict: dict,
    input_dir: str | Path,
    output_dir: str | Path,
    ckpt_filename: str = "best.keras",
    batch_size: int = 32,
    step_seconds: float = 5.0,
    exts: list[str] = None,
    overwrite: bool = False
) -> None:
    """
    Main entry point for directory-level inference.
    """
    if exts is None:
        exts = ["wav", "flac", "mp3"]

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve model path dynamically from config dictionary
    ckpt_dir = config_dict.get("CHECKPOINT_DIR", "")
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    print(f"[INFO] Loading Model from {ckpt_path}...")
    model = load_custom_model(ckpt_path)
    
    print("[INFO] Loading Perch Wrapper...")
    perch = PerchWrapper()
    _ = perch.get_spatial_embedding(np.zeros(5 * 32000, dtype=np.float32)) # Warmup

    audio_files = list_audio_files(input_dir, exts)
    if not audio_files:
        print(f"[WARN] No audio files found in {input_dir}")
        return

    for af in audio_files:
        out_csv = output_dir / (af.stem + ".csv")
        if out_csv.exists() and not overwrite:
            print(f"[SKIP] {af.name} (exists)")
            continue
        try:
            process_file(af, model, perch, out_csv, batch_size, step_seconds, config_dict)
        except Exception as e:
            print(f"[ERROR] Failed processing {af}: {e}")
            import traceback
            traceback.print_exc()