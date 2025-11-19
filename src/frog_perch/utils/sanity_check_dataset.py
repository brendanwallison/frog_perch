import os
import csv
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.training.dataset_builders import (
    build_tf_dataset, build_tf_val_dataset
)
import frog_perch.config as config


# ----------------------------------------------------------------------
# Helper: extract one batch WITH metadata (filename + start sample)
# ----------------------------------------------------------------------

def get_raw_batch(dataset_obj, batch_size=8):
    """
    Returns:
        spatial_batch: (B, H, W, C)
        label_batch: (B, ...)
        file_list: list of strings length B
        start_list: list of ints length B
    """

    batch = []
    for i in range(batch_size):
        spatial, label, audio_file, start = dataset_obj[i]
        batch.append((spatial, label, audio_file, start))

    spatial_batch = np.stack([b[0] for b in batch], axis=0)
    label_batch = [b[1] for b in batch]
    file_list = [b[2] for b in batch]
    start_list = [b[3] for b in batch]

    return spatial_batch, label_batch, file_list, start_list


# ----------------------------------------------------------------------
# Helper: write CSV summary
# ----------------------------------------------------------------------

def write_csv(csv_path, file_list, start_list, label_list, sample_rate):
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "start_sec", "end_sec", "label"])

        for fname, start, label in zip(file_list, start_list, label_list):
            start_sec = start / sample_rate
            end_sec = start_sec + config.CLIP_DURATION_SECONDS
            w.writerow([fname, start_sec, end_sec, label])

    print(f"[OK] Wrote CSV: {csv_path}")


# ----------------------------------------------------------------------
# Helper: plot spectrograms for manual inspection
# ----------------------------------------------------------------------

def save_spectrograms(spatial_batch, filename_batch, out_dir):
    """
    spatial_batch: [B, 1536, 4, 16]
    filename_batch: list of filenames
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    B = spatial_batch.shape[0]

    for i in range(B):
        spatial = spatial_batch[i]  # [1536,4,16]

        # collapse channels → 2D map
        spec2d = spatial.mean(axis=0)   # [4,16]

        plt.imshow(spec2d, aspect='auto', origin='lower')
        plt.colorbar()
        plt.title(filename_batch[i].decode() if isinstance(filename_batch[i], bytes) else filename_batch[i])

        out_path = os.path.join(out_dir, f"{i:04d}.png")
        plt.savefig(out_path)
        plt.close()



# ----------------------------------------------------------------------
# Split leak check
# ----------------------------------------------------------------------

def check_split_leak(train_ds_obj, val_ds_obj):
    train_files = set(train_ds_obj.audio_files)
    val_files = set(val_ds_obj.audio_files)

    leak = train_files & val_files
    if leak:
        print("❌ DATA LEAK DETECTED:", leak)
    else:
        print("✅ Train/test split is clean (no overlapping audio files).")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(batch_size):
    print("\n=== Building dataset objects ===")

    train_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=True,
        pos_ratio=config.POS_RATIO,
        random_seed=config.RANDOM_SEED,
        label_mode='count',
    )

    val_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False,
        pos_ratio=None,
        random_seed=config.RANDOM_SEED,
        label_mode='count',
        val_stride_sec=config.VAL_STRIDE_SEC,
    )

    print("\n=== Checking for split leakage ===")
    check_split_leak(train_obj, val_obj)

    print("\n=== Sampling train batch ===")
    spatial, labels, fnames, starts = get_raw_batch(train_obj, batch_size)
    write_csv("train_batch.csv", fnames, starts, labels, train_obj.sample_rate)
    save_spectrograms(spatial, fnames, "train_specs")

    print("\n=== Sampling validation batch ===")
    spatial_v, labels_v, fnames_v, starts_v = get_raw_batch(val_obj, batch_size)
    write_csv("val_batch.csv", fnames_v, starts_v, labels_v, val_obj.sample_rate)
    save_spectrograms(spatial_v, fnames_v, "val_specs")

    print("\n=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    main(args.batch)
