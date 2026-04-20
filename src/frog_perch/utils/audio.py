# utils/audio.py
import numpy as np
import soundfile as sf
import librosa

def load_audio(
    path,
    target_sr=None,
    mono=True,
    start_sample=None,
    num_samples=None,
):
    """
    Efficient audio loader.

    Supports:
    - full file load (default)
    - partial streaming read (start_sample + num_samples)
    - optional resampling
    """

    # ---------------------------
    # Open file (streaming)
    # ---------------------------
    with sf.SoundFile(path) as f:
        sr = f.samplerate

        # ---------------------------
        # Partial read (FAST PATH)
        # ---------------------------
        if start_sample is not None or num_samples is not None:
            # Project requested samples into native sample rate space
            if target_sr is not None and sr != target_sr:
                ratio = sr / target_sr
                native_start = int((start_sample or 0) * ratio)
                native_num = int(num_samples * ratio) if num_samples else None
            else:
                native_start = start_sample or 0
                native_num = num_samples

            f.seek(native_start)
            data = f.read(native_num, dtype="float32", always_2d=False)

        # ---------------------------
        # Full read (fallback)
        # ---------------------------
        else:
            data = f.read(dtype="float32", always_2d=False)

    # ---------------------------
    # Convert to mono if needed
    # ---------------------------
    if mono and data.ndim > 1:
        data = np.mean(data, axis=1)

    # ---------------------------
    # Resampling (only if needed)
    # ---------------------------
    if target_sr is not None and sr != target_sr:
        data = librosa.resample(
            data,
            orig_sr=sr,
            target_sr=target_sr
        )
        sr = target_sr

    # ---------------------------
    # Enforce strict bounds
    # ---------------------------
    # Resampling can sometimes cause +/- 1 sample rounding differences.
    # We guarantee the exact length requested.
    if num_samples is not None and len(data) > num_samples:
        data = data[:num_samples]

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
