# utils/audio.py
import numpy as np
import soundfile as sf
import librosa

def load_audio(path, target_sr=None, mono=True):
    """
    Load audio file with soundfile, return numpy float32 1-D array and samplerate.
    If target_sr is provided and differs from file sr, resample with librosa.
    """
    data, sr = sf.read(path, always_2d=False)
    if data.ndim > 1 and mono:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    if target_sr is not None and sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return data, sr

def resample_array(arr, orig_sr, target_sr):
    """
    Resample a numpy 1D array from orig_sr to target_sr using librosa.
    """
    if orig_sr == target_sr:
        return arr.astype(np.float32)
    return librosa.resample(arr.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)

def ensure_length(arr, target_len, random_crop=False):
    """
    Ensure arr length equals target_len. If longer, crop; if shorter, pad with zeros.
    random_crop: choose random start when cropping; else start at 0
    """
    L = len(arr)
    if L == target_len:
        return arr
    if L < target_len:
        out = np.zeros(target_len, dtype=np.float32)
        out[:L] = arr
        return out
    # L > target_len
    if random_crop:
        start = np.random.randint(0, L - target_len + 1)
    else:
        start = 0
    return arr[start:start + target_len]
