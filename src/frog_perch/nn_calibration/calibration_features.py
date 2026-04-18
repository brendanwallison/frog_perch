"""
calibration_feature_extraction.py

Extracts ground truth counts and neural network predictions/moments 
from a labeled dataset split for Bayesian calibration.
"""
import os
import numpy as np
import pandas as pd

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.utils.audio import load_audio
from frog_perch.nn_models.model_utils import load_custom_model
from frog_perch.nn_calibration.feature_extraction import (
    calculate_bandpass_features, 
    calculate_window_moments
)

def extract_calibration_features(
    config_dict: dict, 
    ckpt_name: str, 
    split: str = 'test'
) -> pd.DataFrame:
    """
    Processes a dataset split to extract model predictions and ground truth 
    distributions for subsequent calibration model training.
    """
    ckpt_dir = config_dict.get("CHECKPOINT_DIR", "")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    print(f"[INFO] Loading model: {ckpt_path}")
    model = load_custom_model(ckpt_path)
    
    audio_dir = config_dict.get("AUDIO_DIR", "")
    
    dataset_kwargs = dict(
        audio_dir=audio_dir,
        annotation_dir=config_dict.get("ANNOTATION_DIR", ""),
        random_seed=config_dict.get("RANDOM_SEED", 42),
        use_continuous_confidence=config_dict.get("USE_CONTINUOUS_CONFIDENCE", True),
        confidence_params=config_dict.get("CONFIDENCE_PARAMS", {})
    )

    ds_obj = FrogPerchDataset(
        split_type=split, 
        val_stride_sec=config_dict.get("VAL_STRIDE_SEC", 1.0),
        **dataset_kwargs
    )
    
    sr = config_dict.get("DATASET_SAMPLE_RATE", 32000)
    clip_seconds = config_dict.get("CLIP_DURATION_SECONDS", 5.0)
    
    rows = []
    bins = np.arange(17)

    print(f"[INFO] Processing {len(ds_obj)} windows from '{split}' split...")

    for i in range(len(ds_obj)):
        # UPDATED: ds_obj now returns a dictionary of labels as the second element
        emb, labels_dict, audio_file, start_sample = ds_obj[i]
        
        audio_path = os.path.join(audio_dir, audio_file)
        full_data, _ = load_audio(audio_path, target_sr=sr)
        
        window_len = int(clip_seconds * sr)
        audio_window = full_data[start_sample : start_sample + window_len]
        if len(audio_window) < window_len:
            audio_window = np.pad(audio_window, (0, window_len - len(audio_window)))

        # --- Use Shared Core Functions ---
        band_results = calculate_bandpass_features(audio_window, sr)
        
        # UPDATED: model.predict returns a dictionary
        preds_dict = model.predict(emb[np.newaxis, ...], verbose=0)
        nn_mu, nn_var = calculate_window_moments(preds_dict)
        
        # UPDATED: Pull ground truth distribution directly from the dataset labels dictionary
        gt_dist = np.array(labels_dict["count_probs"]).flatten()
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

        if i > 0 and i % 100 == 0: 
            print(f"Processed {i}/{len(ds_obj)}...")

    df = pd.DataFrame(rows)
    save_path = os.path.join(ckpt_dir, f"{ckpt_name}_multiband_calibration.csv")
    df.to_csv(save_path, index=False)
    print(f"[INFO] Saved calibration mapping to: {save_path}")
    
    return df