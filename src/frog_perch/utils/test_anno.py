#!/usr/bin/env python3

from frog_perch.datasets.frog_dataset import FrogPerchDataset

def main():
    ds = FrogPerchDataset(
        train=False,
        val_stride_sec=5.0,
        label_mode="count",
        audio_dir="/home/breallis/datasets/frog_calls/round_2",
        annotation_dir="/home/breallis/datasets/frog_calls/round_2",
        equalize_q2_val=True,
    )

    print("Dataset length:", len(ds))

    # Load item 10
    x, y, audio_file, start_sample = ds[360]

    print("==== Item 10 ====")
    print("audio_file:", audio_file)
    print("start_sample:", start_sample)
    print("x.shape:", getattr(x, "shape", type(x)))
    print("y:", y)

if __name__ == "__main__":
    main()
