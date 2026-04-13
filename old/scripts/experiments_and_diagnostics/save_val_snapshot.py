import sys
import os
import json
import numpy as np

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import frog_perch.config as config
from frog_perch.datasets.frog_dataset import FrogPerchDataset

def save_snapshot():
    output_filename = "validation_set_snapshot.json"
    print(f"--- Generating Validation Snapshot for Seed {config.RANDOM_SEED} ---")
    
    # 1. Instantiate the dataset in Validation Mode
    # We use the exact same logic as train.py to ensure 1:1 correspondence
    ds = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False,
        pos_ratio=None,
        random_seed=config.RANDOM_SEED,
        label_mode=config.LABEL_MODE,
        val_stride_sec=getattr(config, "VAL_STRIDE_SEC", 1.0),
        q2_confidence=getattr(config, "Q2_CONFIDENCE", 0.75),
        equalize_q2_val=getattr(config, "EQUALIZE_Q2_VAL", False),
        use_continuous_confidence=getattr(config, "USE_CONTINUOUS_CONFIDENCE", False),
        confidence_params=getattr(config, "CONFIDENCE_PARAMS", {})
    )

    if not hasattr(ds, 'val_index'):
        print("Error: Dataset object does not have 'val_index'. Is this the correct version of frog_dataset.py?")
        return

    # 2. Extract Metadata
    # We save the config used to generate this, just for record-keeping
    snapshot = {
        "meta": {
            "random_seed": config.RANDOM_SEED,
            "val_stride_sec": ds.val_stride_sec,
            "total_samples": len(ds),
            "generated_from": "FrogPerchDataset logic"
        },
        "index": []
    }

    # 3. Serialize the Index
    # ds.val_index is a list of tuples: (filename, start_time)
    print(f"Extracting {len(ds.val_index)} validation windows...")
    
    for audio_file, start_time in ds.val_index:
        snapshot["index"].append({
            "audio_file": audio_file,
            "start_time_sec": float(start_time) # Ensure it's a standard float, not numpy float
        })

    # 4. Save to Disk
    output_path = os.path.join(os.getcwd(), output_filename)
    with open(output_path, 'w') as f:
        json.dump(snapshot, f, indent=2)

    print(f"\n[Success] Snapshot saved to: {output_path}")
    print("You can now verify this file contains the expected number of items.")

if __name__ == "__main__":
    save_snapshot()