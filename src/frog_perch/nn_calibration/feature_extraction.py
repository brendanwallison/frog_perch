import numpy as np
from scipy.signal import butter, filtfilt

def bandpass_filter(data, sr, low_hz, high_hz, order=4):
    """Applies a Butterworth bandpass filter to a 1D audio array."""
    nyq = 0.5 * sr
    low, high = low_hz / nyq, high_hz / nyq
    if high >= 1.0:
        high = 0.99
    
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, data)

def calculate_bandpass_features(audio_window, sr, sub_win_sec=0.1):
    """Extracts mean, variance, and log-mean RMS for specific frequency bands."""
    sub_win_samples = int(sub_win_sec * sr)
    band_results = {}
    
    for low, high in [(1000, 1500), (1500, 2000)]:
        filtered = bandpass_filter(audio_window, sr, low, high)
        n_chunks = len(filtered) // sub_win_samples
        if n_chunks == 0:
            continue
            
        chunks = filtered[:n_chunks * sub_win_samples].reshape(n_chunks, sub_win_samples)
        rms_per_chunk = np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)
        
        band_key = f"{low}_{high}"
        band_results[f"mean_rms_{band_key}"] = float(np.mean(rms_per_chunk))
        band_results[f"var_rms_{band_key}"] = float(np.var(rms_per_chunk))
        band_results[f"log_mean_rms_{band_key}"] = float(np.log(np.mean(rms_per_chunk) + 1e-12))
        
    return band_results

def calculate_window_moments(predictions_dict):
    """Calculates expected count (mu) and variance (var) directly from count_probs."""
    count_probs = np.asarray(predictions_dict["count_probs"])
    max_bin = count_probs.shape[1] - 1
    bins = np.arange(max_bin + 1)
    
    nn_mu = np.sum(count_probs * bins, axis=1)
    nn_var = np.sum(count_probs * (bins**2), axis=1) - (nn_mu**2)
    return nn_mu, nn_var

def build_feature_record(audio_window, sr, preds_dict_idx):
    """Shared single-window record generator for calibration and inference."""
    band_feats = calculate_bandpass_features(audio_window, sr)
    nn_mu, nn_var = calculate_window_moments(preds_dict_idx)
    
    record = {
        "nn_mu": float(nn_mu[0]),
        "nn_var": float(nn_var[0])
    }
    record.update(band_feats)
    return record

import numpy as np

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    pos = x >= 0
    out = np.empty_like(x, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    out[~pos] = np.exp(x[~pos]) / (1.0 + np.exp(x[~pos]))
    return out

def _flatten_if_needed(x):
    if x is None:
        return None
    a = np.asarray(x)
    if a.ndim == 0:
        return a.reshape(-1)
    if a.ndim > 1:
        # If shape is (batch, n_slices, 1) or (1, n_slices), flatten to (n_slices,)
        if a.shape[0] == 1:
            a = a[0]
        a = a.reshape(-1)
    return a

def decode_nn_outputs(preds_dict):
    """
    Decode model/inspection outputs into a consistent dictionary.

    Returns keys:
      - slice_logits: 1D numpy array (length n_slices) of slice-branch logits
      - slice_probs:  1D numpy array (length n_slices) of slice-branch probabilities
      - count_slice_probs: 1D numpy array (length n_slices) of per-slice probs from count branch (optional)
      - count_probs: 1D numpy array (length max_bin+1) aggregated PMF over counts
      - binary_prob: float or None
      - n_slices: int
      - max_bin: int
    """
    # Accept preds_dict that may be a dict-like mapping or an object with keys
    if not isinstance(preds_dict, dict):
        raise KeyError(f"decode_nn_outputs expects a dict-like preds_dict; got {type(preds_dict)}")

    # -----------------------------
    # SLICE logits / probs
    # -----------------------------
    slice_logits = None
    # Prefer explicit 'slice' (training callback used preds['slice'] as flattened logits)
    if "slice" in preds_dict and preds_dict["slice"] is not None:
        slice_logits = _flatten_if_needed(preds_dict["slice"])
    # Fallback to 'slice_logits' if present
    elif "slice_logits" in preds_dict and preds_dict["slice_logits"] is not None:
        slice_logits = _flatten_if_needed(preds_dict["slice_logits"])
    else:
        raise KeyError(f"Missing slice output. Available keys: {list(preds_dict.keys())}")

    # -----------------------------
    # COUNT aggregated PMF
    # -----------------------------
    if "count_probs" not in preds_dict or preds_dict["count_probs"] is None:
        raise KeyError(f"Missing 'count_probs'. Available keys: {list(preds_dict.keys())}")
    count_probs = _flatten_if_needed(preds_dict["count_probs"])

    # -----------------------------
    # OPTIONAL: per-slice count probs (from count branch before aggregation)
    # -----------------------------
    count_slice_probs = None
    if "count_slice_probs" in preds_dict and preds_dict["count_slice_probs"] is not None:
        # This is expected to already be probabilities (sigmoid applied in model)
        count_slice_probs = _flatten_if_needed(preds_dict["count_slice_probs"])
        # If values look like logits (outside [0,1]) convert to probs defensively
        if np.any(count_slice_probs < 0) or np.any(count_slice_probs > 1):
            count_slice_probs = _sigmoid(count_slice_probs)

    # -----------------------------
    # OPTIONAL: binary
    # -----------------------------
    binary_prob = preds_dict.get("binary", None)
    if binary_prob is not None:
        bp = np.asarray(binary_prob)
        if bp.ndim > 0:
            try:
                binary_prob = float(bp.reshape(-1)[0])
            except Exception:
                binary_prob = None
        else:
            try:
                binary_prob = float(bp)
            except Exception:
                binary_prob = None

    # -----------------------------
    # Derive slice_probs from slice_logits (sigmoid)
    # -----------------------------
    slice_logits = np.asarray(slice_logits, dtype=float)
    slice_probs = _sigmoid(slice_logits)

    # -----------------------------
    # Metadata
    # -----------------------------
    n_slices = int(slice_logits.shape[0])
    max_bin = int(count_probs.shape[-1] - 1)

    return {
        "slice_logits": slice_logits,
        "slice_probs": slice_probs,
        "count_slice_probs": count_slice_probs,
        "count_probs": count_probs,
        "binary_prob": binary_prob,
        "n_slices": n_slices,
        "max_bin": max_bin,
    }
