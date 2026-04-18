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

def calculate_window_moments(predictions_dict):
    """
    Takes the dictionary output from the new Keras model and calculates 
    the expected count (mu) and variance (var) directly from the count_probs.
    """
    # predictions_dict["count_probs"] shape: [B, max_bin + 1]
    count_probs = np.asarray(predictions_dict["count_probs"])
    
    # Create an array of bins [0, 1, 2, ..., max_bin]
    max_bin = count_probs.shape[1] - 1
    bins = np.arange(max_bin + 1)
    
    # E[X] = sum(k * P(k))
    nn_mu = np.sum(count_probs * bins, axis=1)
    
    # Var(X) = E[X^2] - (E[X])^2
    nn_var = np.sum(count_probs * (bins**2), axis=1) - (nn_mu**2)
    
    return nn_mu, nn_var