#!/usr/bin/env python3
"""
infer_slice_directory.py

Inference for the new model version with linear interpolation and an output-resolution toggle.

Usage highlights:
  --step_seconds    : step size for sliding windows and output grid (must be <= CLIP_DURATION_SECONDS)
  --out_resolution : "native" (5/16s) or "step" (same as step_seconds)
  --ckpt, --input_dir, --output_dir, --batch_size, --ext, --overwrite

Outputs CSV per audio file with columns:
  time_sec, prob, n_eff

Each slice probability is linearly interpolated to the two nearest grid bins (left/right).
This version computes a weighted effective-sample-size per grid cell and writes it as n_eff.
"""
import argparse
import csv
import sys
from pathlib import Path
import math
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
import experiments.infer_slice_config as cfg

# -------------------------
# Utilities
# -------------------------
def list_audio_files(root: Path, exts):
    files = []
    for p in sorted(root.iterdir()):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files


def windows_for_file(total_seconds, clip_seconds, stride):
    max_start = max(0.0, total_seconds - clip_seconds)
    if max_start < 0:
        return np.array([], dtype=float)
    # include last start at max_start if it aligns
    return np.arange(0.0, max_start + 1e-8, stride)


def load_model_for_slices(ckpt_path: str):
    """
    Load model and return (model_for_pred, logits_layer_name_or_None).
    The model you showed returns slice_logits_flat as the model output with shape [B, T].
    We prefer a named logits layer if present, otherwise use model.output.
    """
    print(f"[INFO] Loading model: {ckpt_path}")
    model = tf.keras.models.load_model(ckpt_path, compile=False)

    # Try to find a named logits layer commonly used
    for name in ("slice_logits_flat", "slice_logits", "logits", "count_logits"):
        try:
            layer = model.get_layer(name)
            print(f"[INFO] Found logits layer: {name}")
            # Build a model that returns that layer's output (logits)
            model_for_pred = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
            return model_for_pred, name
        except Exception:
            continue

    # Fallback: model.output is logits (your return shows model returns slice_logits)
    print("[INFO] No named logits layer found; using model.output as logits")
    return model, None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensure_1d_probs(arr):
    """
    Accept logits or probs shaped [B, T] or [B, T, 1].
    Convert logits -> sigmoid if values outside [0,1] (with small epsilon tolerance).
    Return shape [B, T] with values in [0,1].
    """
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"Unexpected prediction shape {a.shape}; expected [B, T] or [B, T, 1]")
    # If values outside [0,1] (with tolerance), treat as logits
    eps = 1e-6
    if np.any(a < -eps) or np.any(a > 1.0 + eps):
        a = sigmoid(a)
    # Clip to [0,1] to avoid tiny numerical issues
    a = np.clip(a, 0.0, 1.0)
    return a


# -------------------------
# Main processing for a single file
# -------------------------
def process_file(path_audio, model, perch, out_csv, batch_size, step_seconds, out_resolution):
    sample_rate = config.DATASET_SAMPLE_RATE
    clip_seconds = config.CLIP_DURATION_SECONDS
    clip_samples = int(sample_rate * clip_seconds)
    perch_sr = config.PERCH_SAMPLE_RATE
    perch_samples = config.PERCH_CLIP_SAMPLES

    # Safety: step_seconds must not exceed clip_seconds
    if step_seconds > clip_seconds:
        raise ValueError(f"step_seconds ({step_seconds}) cannot be larger than clip duration ({clip_seconds})")

    # native slice duration (5/16 s) derived from TIME_SLICES if available
    TIME_SLICES = getattr(config, "TIME_SLICES", 16)
    native_slice_duration = clip_seconds / float(TIME_SLICES)  # typically 5 / 16 = 0.3125s

    print(f"[INFO] Processing {path_audio.name} (step={step_seconds}s, out_res={out_resolution})")
    audio, sr = load_audio(str(path_audio), target_sr=sample_rate)
    total_seconds = len(audio) / sample_rate

    starts = windows_for_file(total_seconds, clip_seconds, stride=step_seconds)
    if len(starts) == 0:
        print(f"[INFO] No windows (file < clip length): {path_audio.name}")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["time_sec", "prob", "n_eff", "var"])
        return

    # Determine output grid and offset (alignment)
    if out_resolution == "native":
        grid_step = native_slice_duration
        # Align grid so that slice centers (which are at (i+0.5)*slice_duration relative to window)
        # fall exactly on grid points: offset by half a slice.
        grid_offset = native_slice_duration / 2.0
    else:  # "step"
        grid_step = step_seconds
        grid_offset = 0.0

    # Build grid times from grid_offset to cover the file
    n_grid = int(math.ceil((total_seconds - grid_offset) / grid_step)) + 1
    grid_times = grid_offset + np.arange(n_grid) * grid_step

    # Accumulators:
    # sum_wx  = sum(w * p)
    # sum_w   = sum(w)
    # sum_wx2 = sum(w * p^2)
    # sum_w2  = sum(w^2)
    sum_wx = np.zeros(n_grid, dtype=np.float64)
    sum_w = np.zeros(n_grid, dtype=np.float64)
    sum_wx2 = np.zeros(n_grid, dtype=np.float64)
    sum_w2 = np.zeros(n_grid, dtype=np.float64)

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
                clip[:val] = audio[st_samp:st_samp + val]

        # resample to Perch
        clip_p = resample_array(clip, sample_rate, perch_sr)
        if len(clip_p) < perch_samples:
            clip_p = np.pad(clip_p, (0, perch_samples - len(clip_p)))
        clip_p = clip_p[:perch_samples]

        embedding = perch.get_spatial_embedding(clip_p).astype(np.float32)

        emb_batch.append(embedding)
        meta_batch.append((idx, float(st), float(st + clip_seconds)))

        if len(emb_batch) >= batch_size:
            _process_batch_interp(emb_batch, meta_batch, model,
                                  grid_times, sum_wx, sum_w, sum_wx2, sum_w2,
                                  clip_seconds, TIME_SLICES, grid_step, grid_offset)
            emb_batch, meta_batch = [], []

    # flush last batch
    if emb_batch:
        _process_batch_interp(emb_batch, meta_batch, model,
                              grid_times, sum_wx, sum_w, sum_wx2, sum_w2,
                              clip_seconds, TIME_SLICES, grid_step, grid_offset)

    # finalize: compute averaged probabilities, effective sample size, and variance
    probs = np.full_like(sum_wx, np.nan, dtype=np.float32)
    n_eff = np.zeros_like(sum_w, dtype=np.float32)
    var_mean = np.full_like(sum_wx, np.nan, dtype=np.float32)

    nonzero = sum_w > 0.0
    if np.any(nonzero):
        pbar = np.zeros_like(sum_wx, dtype=np.float64)
        pbar[nonzero] = sum_wx[nonzero] / sum_w[nonzero]
        # weighted population variance: var_w = (sum_wx2 / W) - pbar^2
        var_w = np.zeros_like(sum_wx, dtype=np.float64)
        var_w[nonzero] = (sum_wx2[nonzero] / sum_w[nonzero]) - (pbar[nonzero] ** 2)
        var_w = np.maximum(var_w, 0.0)
        # effective sample size: N_eff = W^2 / sum(w^2)
        n_eff_vals = np.zeros_like(sum_w, dtype=np.float64)
        mask_nonzero = nonzero & (sum_w2 > 0.0)
        n_eff_vals[mask_nonzero] = (sum_w[mask_nonzero] ** 2) / sum_w2[mask_nonzero]
        # variance of weighted mean approx: var_mean = var_w / N_eff
        var_mean_vals = np.zeros_like(var_w, dtype=np.float64)
        mask = mask_nonzero & (n_eff_vals > 0.0)
        var_mean_vals[mask] = var_w[mask] / n_eff_vals[mask]
        # final outputs
        probs[nonzero] = pbar[nonzero].astype(np.float32)
        n_eff[nonzero] = n_eff_vals[nonzero].astype(np.float32)
        var_mean[nonzero] = var_mean_vals[nonzero].astype(np.float32)

    # write CSV (time_sec, prob, n_eff, var)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_sec", "prob", "n_eff", "var"])
        for t, p, ne, v in zip(grid_times, probs, n_eff, var_mean):
            p_out = "" if np.isnan(p) else float(p)
            ne_out = 0.0 if (ne is None or np.isnan(ne)) else float(ne)
            v_out = "" if np.isnan(v) else float(v)
            w.writerow([float(t), p_out, ne_out, v_out])

    print(f"[INFO] Wrote {len(grid_times)} rows → {out_csv}")


def _process_batch_interp(emb_batch, meta_batch, model,
                          grid_times, sum_wx, sum_w, sum_wx2, sum_w2,
                          clip_seconds, time_slices, grid_step, grid_offset):
    """
    Run model on batch and linearly interpolate each slice probability to the two nearest
    grid bins (left/right) using distance weights, while accumulating:
      - sum_wx  (sum of w * p)
      - sum_w   (sum of w)
      - sum_wx2 (sum of w * p^2)
      - sum_w2  (sum of w^2)

    Note: float index is computed as (center - grid_offset) / grid_step so the grid origin
    (grid_offset) is respected and alignment is purely a grid definition.
    """
    X = np.stack(emb_batch)
    # model.predict may return logits or a tuple; handle both
    preds = model.predict(X, verbose=0)
    if isinstance(preds, (list, tuple)):
        logits = preds[0]
    else:
        logits = preds
    probs_batch = ensure_1d_probs(logits)  # shape [B, T]
    B, T = probs_batch.shape
    slice_duration = clip_seconds / float(T)

    # For each clip, compute slice centers and distribute each slice prob to two nearest grid bins
    for (idx, st, ed), probs in zip(meta_batch, probs_batch):
        centers = st + (np.arange(T, dtype=float) + 0.5) * slice_duration  # center times
        # For each center, compute floating grid index = (center - grid_offset) / grid_step
        float_idxs = (centers - grid_offset) / grid_step
        left_idxs = np.floor(float_idxs).astype(int)
        right_idxs = left_idxs + 1
        # weights: right_weight = frac, left_weight = 1 - frac
        frac = float_idxs - left_idxs
        left_w = 1.0 - frac
        right_w = frac

        # Clip indices to grid bounds and adjust weights for edge cases
        for li, ri, lw, rw, p in zip(left_idxs, right_idxs, left_w, right_w, probs):
            # If both indices are outside grid, skip
            if ri < 0 or li >= len(grid_times):
                continue
            # left contribution
            if 0 <= li < len(grid_times) and lw > 0.0:
                sum_wx[li] += float(p) * float(lw)
                sum_w[li] += float(lw)
                sum_wx2[li] += float((p * p) * lw)
                sum_w2[li] += float(lw * lw)
            # right contribution
            if 0 <= ri < len(grid_times) and rw > 0.0:
                sum_wx[ri] += float(p) * float(rw)
                sum_w[ri] += float(rw)
                sum_wx2[ri] += float((p * p) * rw)
                sum_w2[ri] += float(rw * rw)


# -------------------------
# CLI wrapper
# -------------------------
def main():
    P = argparse.ArgumentParser(add_help=False)
    P.add_argument("--ckpt")
    P.add_argument("--input_dir")
    P.add_argument("--output_dir")
    P.add_argument("--batch_size", type=int)
    P.add_argument("--step_seconds", type=float, help="grid step size in seconds (must be <= clip length)")
    P.add_argument("--out_resolution", choices=("native", "step"), default=None,
                   help="Output resolution: 'native' (slice duration) or 'step' (same as step_seconds)")
    P.add_argument("--ext")
    P.add_argument("--overwrite", action="store_true")
    args = P.parse_args()

    ckpt = args.ckpt or cfg.CKPT
    input_dir = Path(args.input_dir or cfg.INPUT_DIR)
    output_dir = Path(args.output_dir or cfg.OUTPUT_DIR)
    batch_size = args.batch_size or cfg.BATCH_SIZE
    step_seconds = args.step_seconds or cfg.STRIDE or (config.CLIP_DURATION_SECONDS / getattr(config, "TIME_SLICES", 16))
    out_resolution = args.out_resolution or cfg.OUT_RESOLUTION or "step"
    exts = (args.ext.split(",") if args.ext else cfg.EXT)
    overwrite = args.overwrite or cfg.OVERWRITE

    # Safety: do not allow step_seconds > clip length
    if step_seconds > config.CLIP_DURATION_SECONDS:
        raise ValueError(f"step_seconds ({step_seconds}) cannot exceed clip duration ({config.CLIP_DURATION_SECONDS})")

    output_dir.mkdir(parents=True, exist_ok=True)

    model, logits_layer_name = load_model_for_slices(ckpt)
    perch = PerchWrapper()

    # quick smoke test of Perch
    _ = perch.get_spatial_embedding(np.random.uniform(-1, 1, 5 * 32000).astype(np.float32))

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
            process_file(af, model, perch, out_csv, batch_size, step_seconds, out_resolution)
        except Exception as e:
            print(f"[ERROR] Failed processing {af}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()