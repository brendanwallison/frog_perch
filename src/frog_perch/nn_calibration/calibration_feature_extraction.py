import os
import numpy as np
import pandas as pd

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.utils.audio import load_audio
from frog_perch.nn_models.model_utils import load_custom_model
from frog_perch.nn_calibration.feature_extraction import build_feature_record

def extract_calibration_features(config_dict: dict, ckpt_name: str, split: str = 'test') -> pd.DataFrame:
    """Extracts ground truth counts and moments from labeled data splits."""
    ckpt_dir = config_dict.get("CHECKPOINT_DIR", "")
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    model = load_custom_model(ckpt_path)
    
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
        use_continuous_confidence=config_dict.get("USE_CONTINUOUS_CONFIDENCE", True)
    )

    ds_obj = FrogPerchDataset(split_type=split, val_stride_sec=config_dict.get("VAL_STRIDE_SEC", 1.0), **dataset_kwargs)
    rows = []

    print(f"[INFO] Running calibration extraction on '{split}' split...")
    for i in range(len(ds_obj)):
        emb, labels_dict, audio_file, start_sample, _ = ds_obj[i]
        
        # Audio framing
        full_data, _ = load_audio(os.path.join(audio_dir, audio_file), target_sr=sr)
        audio_window = full_data[start_sample : start_sample + window_len]
        if len(audio_window) < window_len:
            audio_window = np.pad(audio_window, (0, window_len - len(audio_window)))

        # Inference and metrics processing
        preds_dict = model.predict(emb[np.newaxis, ...], verbose=0)
        res = build_feature_record(audio_window, sr, preds_dict)
        
        # Ground-truth distribution parsing
        gt_dist = np.array(labels_dict["count_probs"]).flatten()
        gt_mu = np.sum(gt_dist * bins)
        
        res.update({
            "file": audio_file,
            "offset": start_sample,
            "gt_mu": gt_mu,
            "gt_var": np.sum(gt_dist * (bins**2)) - (gt_mu**2),
            "q_k": gt_dist.tolist(),
        })
        rows.append(res)

    df = pd.DataFrame(rows)
    save_path = os.path.join(ckpt_dir, f"{ckpt_name}_multiband_calibration.csv")
    df.to_csv(save_path, index=False)
    return df