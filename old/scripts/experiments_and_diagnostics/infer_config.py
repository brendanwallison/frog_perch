# infer_config.py
"""
Default configuration for infer_directory.py.
Edit these values during debugging so you never need CLI args.
"""

CKPT = "/home/breallis/dev/frog_perch/checkpoints/pool=mlp_flat_loss=count_x0=-4.0_k=1.0.keras"

INPUT_DIR = "/home/breallis/datasets/frog_calls/gabon_full/P2"
OUTPUT_DIR = "/home/breallis/datasets/frog_calls/gabon_full/P2_out"

BATCH_SIZE = 64
STRIDE = 5.0
EXT = ["wav", "flac"]

OVERWRITE = False
