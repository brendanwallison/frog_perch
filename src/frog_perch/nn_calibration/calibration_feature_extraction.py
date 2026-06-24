#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.utils.audio import load_audio
from frog_perch.nn_models.model_utils import load_custom_model
from frog_perch.nn_calibration.feature_extraction import build_feature_record
from frog_perch.nn_calibration.feature_extraction import decode_nn_outputs


def pack(x: Optional[Any]) -> Optional[str]:
    """Serialize numpy arrays or iterables to JSON for CSV storage."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return json.dumps(x.tolist())
    try:
        return json.dumps(list(x))
    except Exception:
        return json.dumps(x)


def _get_layer_output_safe(m: tf.keras.Model, name: str):
    """Return layer.output if layer exists, else None."""
    try:
        return m.get_layer(name).output
    except Exception:
        return None


def _flatten_if_needed(arr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if arr is None:
        return None
    a = np.asarray(arr)
    if a.ndim > 1:
        return a.reshape(-1)
    return a


def _ensure_batch_dim(x: Optional[Any]) -> Optional[np.ndarray]:
    """Ensure array has a leading batch dimension (1, ...)."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 1:
        return a[np.newaxis, ...]
    if a.ndim == 0:
        return a.reshape(1, 1)
    return a


def extract_calibration_features(config_dict: dict, ckpt_name: str, split: str = "test") -> pd.DataFrame:
    """Extracts ground truth counts and moments from labeled data splits.

    Builds an inspection model to expose per-slice outputs from the count branch
    and saves them (nn_count_slice_probs) along with slice logits (nn_slice_logits).
    Ensures arrays passed into build_feature_record have expected batch dims.
    """
    ckpt_dir = config_dict.get("CHECKPOINT_DIR", "")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model = load_custom_model(ckpt_path)

    # Build an inspection model that exposes intermediate tensors if available.
    slice_out = _get_layer_output_safe(model, "slice")  # flattened slice logits
    count_slice_probs_out = _get_layer_output_safe(model, "count_slice_probs")  # per-slice sigmoid output
    count_probs_out = _get_layer_output_safe(model, "count_probs")  # aggregated PMF
    binary_out = _get_layer_output_safe(model, "binary")

    from tensorflow.keras.layers import Flatten

    if count_slice_probs_out is not None and len(count_slice_probs_out.shape) == 3 and count_slice_probs_out.shape[-1] == 1:
        count_slice_probs_flat = Flatten()(count_slice_probs_out)
    else:
        count_slice_probs_flat = count_slice_probs_out

    if slice_out is not None and len(slice_out.shape) == 3 and slice_out.shape[-1] == 1:
        slice_out_flat = Flatten()(slice_out)
    else:
        slice_out_flat = slice_out

    inspection_outputs: Dict[str, Any] = {}
    if slice_out_flat is not None:
        inspection_outputs["slice_logits"] = slice_out_flat
    if count_slice_probs_flat is not None:
        inspection_outputs["count_slice_probs"] = count_slice_probs_flat
    if count_probs_out is not None:
        inspection_outputs["count_probs"] = count_probs_out
    if binary_out is not None:
        inspection_outputs["binary"] = binary_out

    if len(inspection_outputs) == 0:
        inspection_model = model
        inspection_outputs = {}
    else:
        inspection_model = tf.keras.Model(inputs=model.inputs, outputs=inspection_outputs, name="inspection_model")

    audio_dir = config_dict.get("AUDIO_DIR", "")
    max_bin = config_dict.get("MAX_BIN", 8)
    sr = config_dict.get("DATASET_SAMPLE_RATE", 32000)
    window_len = int(config_dict.get("CLIP_DURATION_SECONDS", 5.0) * sr)
    bins = np.arange(max_bin + 1)

    dataset_kwargs = dict(
        audio_dir=audio_dir,
        annotation_dir=config_dict.get("ANNOTATION_DIR", ""),
        random_seed=config_dict.get("RANDOM_SEED", 42),
        confidence_params=config_dict.get("CONFIDENCE_PARAMS", {}),
        n_slices=config_dict.get("N_SLICES", 16),
        max_bin=max_bin,
        use_continuous_confidence=config_dict.get("USE_CONTINUOUS_CONFIDENCE", True),
    )

    ds_obj = FrogPerchDataset(
        split_type=split, val_stride_sec=config_dict.get("VAL_STRIDE_SEC", 1.0), **dataset_kwargs
    )
    rows = []
    rows_full = []

    print(f"[INFO] Running calibration extraction on '{split}' split...")

    for i in range(len(ds_obj)):
        emb, labels_dict, audio_file, start_sample, events = ds_obj[i]

        # Audio framing
        full_data, _ = load_audio(os.path.join(audio_dir, audio_file), target_sr=sr)
        audio_window = full_data[start_sample : start_sample + window_len]
        if len(audio_window) < window_len:
            audio_window = np.pad(audio_window, (0, window_len - len(audio_window)))

        # Inference using inspection model (may return dict of arrays)
        preds_raw = inspection_model.predict(emb[np.newaxis, ...], verbose=0)

        # Normalize preds_raw to dict keyed by inspection_outputs order if necessary
        if isinstance(preds_raw, dict):
            preds_dict = preds_raw
        else:
            keys = list(inspection_outputs.keys())
            if len(keys) == 0:
                preds_dict = {"model_output": preds_raw}
            else:
                preds_dict = {k: preds_raw[idx] for idx, k in enumerate(keys)}

        # Build preds_for_decode with expected batch dims for downstream utilities
        preds_for_decode: Dict[str, Any] = {}

        # slice: prefer 'slice' or 'slice_logits'
        if "slice" in preds_dict and preds_dict["slice"] is not None:
            preds_for_decode["slice"] = _ensure_batch_dim(preds_dict["slice"])
        elif "slice_logits" in preds_dict and preds_dict["slice_logits"] is not None:
            preds_for_decode["slice"] = _ensure_batch_dim(preds_dict["slice_logits"])

        # count_probs: aggregated PMF must have batch dim
        if "count_probs" in preds_dict and preds_dict["count_probs"] is not None:
            preds_for_decode["count_probs"] = _ensure_batch_dim(preds_dict["count_probs"])

        # binary if present
        if "binary" in preds_dict and preds_dict["binary"] is not None:
            preds_for_decode["binary"] = preds_dict["binary"]

        # Build feature record using preds_for_decode (keeps previous behavior)
        res = build_feature_record(audio_window, sr, preds_for_decode)
        decoded = decode_nn_outputs(preds_for_decode)

        # Ground-truth distribution parsing
        gt_dist = np.array(labels_dict["count_probs"]).flatten()
        gt_mu = np.sum(gt_dist * bins)
        gt_var = np.sum(gt_dist * (bins**2)) - (gt_mu**2)
        gt_slice = np.asarray(labels_dict["slice"]).flatten()

        # Prepare full row (model outputs + metadata)
        res_full = dict(res)

        # Extract and flatten inspection outputs for saving
        slice_logits_arr = _flatten_if_needed(preds_dict.get("slice_logits"))
        count_slice_probs_arr = _flatten_if_needed(preds_dict.get("count_slice_probs"))
        count_probs_arr = _flatten_if_needed(preds_dict.get("count_probs"))
        binary_arr = preds_dict.get("binary", None)
        binary_val = None
        if binary_arr is not None:
            try:
                binary_val = float(np.asarray(binary_arr).reshape(-1)[0])
            except Exception:
                binary_val = None

        # Populate res_full with both per-slice and aggregated outputs
        res_full.update(
            {
                "nn_slice_logits": pack(slice_logits_arr),
                "nn_count_slice_probs": pack(count_slice_probs_arr),
                "nn_count_probs": pack(count_probs_arr),
                "nn_binary": binary_val,
                "n_slices": int(decoded.get("n_slices")) if decoded.get("n_slices") is not None else config_dict.get("N_SLICES"),
                "max_bin": int(decoded.get("max_bin")) if decoded.get("max_bin") is not None else max_bin,
            }
        )

        # Serialize events as JSON with plain Python lists so CSV stores a JSON string
        events_serializable = None
        if events is not None:
            events_serializable = {
                "starts": list(np.asarray(events.get("starts")).tolist()) if events.get("starts") is not None else [],
                "ends": list(np.asarray(events.get("ends")).tolist()) if events.get("ends") is not None else [],
                "conf": list(np.asarray(events.get("conf")).tolist()) if events.get("conf") is not None else [],
            }

        res_full.update(
            {
                "file": audio_file,
                "offset": int(start_sample),
                "gt_mu": float(gt_mu),
                "gt_var": float(gt_var),
                "gt_slice": pack(gt_slice),
                "q_k": gt_dist.tolist(),
                "events": json.dumps(events_serializable) if events_serializable is not None else None,
                "dataset_idx": int(i),
            }
        )

        # A lighter row (if you want a smaller CSV) — include key metadata and aggregated PMF
        res.update(
            {
                "file": audio_file,
                "offset": int(start_sample),
                "gt_mu": float(gt_mu),
                "gt_var": float(gt_var),
                "q_k": gt_dist.tolist(),
                "events": json.dumps(events_serializable) if events_serializable is not None else None,
                "dataset_idx": int(i),
            }
        )

        rows.append(res)
        rows_full.append(res_full)

    df = pd.DataFrame(rows)
    save_path = os.path.join(ckpt_dir, f"{ckpt_name}_multiband_calibration.csv")
    df.to_csv(save_path, index=False)
    df_full = pd.DataFrame(rows_full)
    save_path_full = os.path.join(ckpt_dir, f"{ckpt_name}_multiband_calibration_full.csv")
    df_full.to_csv(save_path_full, index=False)
    return df, df_full
