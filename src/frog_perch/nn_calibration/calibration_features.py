import os
import numpy as np
import pandas as pd
import frog_perch.config as config
from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.metrics.slice_to_count_metrics import targets_to_soft_counts

# Import from the new core module
from frog_perch.nn_calibration.feature_extraction import (
    calculate_bandpass_features, 
    load_custom_model, 
    ensure_1d_probs, 
    calculate_window_moments
)
from frog_perch.utils.audio import load_audio

def run_calibration_extraction(ckpt_name, split='test'):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, ckpt_name)
    print(f"Loading model: {ckpt_path}")
    model = load_custom_model(ckpt_path)
    
    dataset_kwargs = dict(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        random_seed=config.RANDOM_SEED,
        label_mode='slice',
        q2_confidence=config.Q2_CONFIDENCE,
        use_continuous_confidence=getattr(config, "USE_CONTINUOUS_CONFIDENCE", True),
        confidence_params=getattr(config, "CONFIDENCE_PARAMS", {})
    )

    ds_obj = FrogPerchDataset(
        split_type=split, 
        val_stride_sec=getattr(config, "VAL_STRIDE_SEC", 1.0),
        **dataset_kwargs
    )
    
    sr = config.DATASET_SAMPLE_RATE 
    rows = []
    bins = np.arange(17)

    print(f"Processing {len(ds_obj)} windows...")

    for i in range(len(ds_obj)):
        emb, gt_slices, audio_file, start_sample = ds_obj[i]
        
        audio_path = os.path.join(config.AUDIO_DIR, audio_file)
        full_data, _ = load_audio(audio_path, target_sr=sr)
        
        window_len = int(config.CLIP_DURATION_SECONDS * sr)
        audio_window = full_data[start_sample : start_sample + window_len]
        if len(audio_window) < window_len:
            audio_window = np.pad(audio_window, (0, window_len - len(audio_window)))

        # --- Use Shared Core Functions ---
        band_results = calculate_bandpass_features(audio_window, sr)
        
        logits = model.predict(emb[np.newaxis, ...], verbose=0)
        probs_batch = ensure_1d_probs(logits)
        
        nn_mu, nn_var = calculate_window_moments(probs_batch)
        
        # Calibration specific logic (Ground Truth)
        gt_dist = targets_to_soft_counts(gt_slices[np.newaxis, ...], max_bin=16).numpy().flatten()
        gt_mu = np.sum(gt_dist * bins)
        gt_var = np.sum(gt_dist * (bins**2)) - (gt_mu**2)

        res = {
            "file": audio_file,
            "offset": start_sample,
            "gt_mu": gt_mu,
            "gt_var": gt_var,
            "nn_mu": nn_mu[0],
            "nn_var": nn_var[0],
            "q_k": gt_dist.tolist(),
        }
        res.update(band_results)
        rows.append(res)

        if i % 100 == 0: print(f"Processed {i}/{len(ds_obj)}...")

    df = pd.DataFrame(rows)
    save_path = os.path.join(config.CHECKPOINT_DIR, f"{ckpt_name}_multiband_calibration.csv")
    df.to_csv(save_path, index=False)
    print(f"Saved to: {save_path}")
    return df

if __name__ == "__main__":
    run_calibration_extraction("pool=slice_loss=slice_x0=-3.0_k=1.0.keras")