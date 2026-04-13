# experiments/infer_config.py
# Placeholder configuration for infer_directory_new_interp.py
# Edit these values or override via CLI arguments.

# Path to the saved Keras model checkpoint (string or Path)
CKPT = "/home/breallis/dev/frog_perch/checkpoints/pool=slice_loss=slice_x0=-4.0_k=1.0.keras"  # e.g., "models/perch_ckpt"

# Input directory containing audio files to process
INPUT_DIR = "/home/breallis/datasets/frog_calls/gabon_full/P2"

# Output directory where per-file CSVs will be written
OUTPUT_DIR = "/home/breallis/datasets/frog_calls/gabon_full/P2_slice_out"

# Batch size for model.predict
BATCH_SIZE = 16

# Default stride / step in seconds for sliding windows.
# If None, the script will fall back to clip duration / TIME_SLICES.
STRIDE = None  # e.g., 0.3125 for native 5/16s; set to None to use default

# Output resolution toggle: "native" or "step"
OUT_RESOLUTION = "native"  # or "native"

# Accepted audio file extensions (list of strings)
EXT = ["wav", "flac", "mp3"]

# If True, overwrite existing CSVs in the output directory
OVERWRITE = False
