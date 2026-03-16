import numpy as np
import tensorflow as tf
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
    """
    Extracts mean, variance, and log-mean RMS for specific frequency bands.
    Returns a dictionary of features.
    """
    sub_win_samples = int(sub_win_sec * sr)
    band_results = {}
    
    for low, high in [(1000, 1500), (1500, 2000)]:
        filtered = bandpass_filter(audio_window, sr, low, high)
        
        # Reshape to calculate temporal variance
        n_chunks = len(filtered) // sub_win_samples
        if n_chunks == 0:
            continue # Window too short
            
        chunks = filtered[:n_chunks * sub_win_samples].reshape(n_chunks, sub_win_samples)
        
        # Calculate RMS per chunk
        rms_per_chunk = np.sqrt(np.mean(chunks**2, axis=1) + 1e-12)
        
        band_key = f"{low}_{high}"
        band_results[f"mean_rms_{band_key}"] = float(np.mean(rms_per_chunk))
        band_results[f"var_rms_{band_key}"] = float(np.var(rms_per_chunk))
        band_results[f"log_mean_rms_{band_key}"] = float(np.log(np.mean(rms_per_chunk) + 1e-12))
        
    return band_results

def ensure_1d_probs(arr):
    """Converts raw NN output to valid probabilities on [0, 1]."""
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"Unexpected prediction shape {a.shape}; expected [B, T] or [B, T, 1]")
    
    eps = 1e-6
    if np.any(a < -eps) or np.any(a > 1.0 + eps):
        a = 1.0 / (1.0 + np.exp(-a)) # Sigmoid
    return np.clip(a, 0.0, 1.0)

def calculate_window_moments(probs_batch):
    """
    Takes [B, T] slice probabilities and returns expected count (mu) 
    and variance across slices (var) for each window.
    """
    # Assuming the NN outputs probabilities for individual slices (e.g., T=16)
    # The expected count is simply the sum of the slice probabilities
    nn_mu = np.sum(probs_batch, axis=1)
    
    # The variance of the sum of independent Bernoulli trials is sum(p * (1-p))
    nn_var = np.sum(probs_batch * (1 - probs_batch), axis=1)
    
    return nn_mu, nn_var

def load_custom_model(ckpt_path):
    """Loads a Keras model with all custom metrics and layers defined."""
    from frog_perch.metrics.slice_to_count_metrics import (
        SliceLossWithSoftCountKL, SliceToCountKLDivergence, 
        SliceExpectedCountMAE, SliceEMD, SliceExpectedRecall, 
        SliceExpectedPrecision, SliceExpectedBinaryAccuracy,
        SliceToCountWrapper
    )
    from frog_perch.metrics.count_metrics import (
        ExpectedRecall, ExpectedPrecision, ExpectedBinaryAccuracy, 
        EMD, ExpectedCountMAE
    )

    custom_objects = {
        "SliceToCountWrapper": SliceToCountWrapper,
        "SliceLossWithSoftCountKL": SliceLossWithSoftCountKL,
        "SliceToCountKLDivergence": SliceToCountKLDivergence,
        "SliceExpectedCountMAE": SliceExpectedCountMAE,
        "SliceEMD": SliceEMD,
        "SliceExpectedRecall": SliceExpectedRecall,
        "SliceExpectedPrecision": SliceExpectedPrecision,
        "SliceExpectedBinaryAccuracy": SliceExpectedBinaryAccuracy,
        "ExpectedCountMAE": ExpectedCountMAE,
        "EMD": EMD,
        "ExpectedRecall": ExpectedRecall,
        "ExpectedPrecision": ExpectedPrecision,
        "ExpectedBinaryAccuracy": ExpectedBinaryAccuracy,
    }
    return tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)