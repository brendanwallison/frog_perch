# config.py
import os

# Paths: set these to your local environment
AUDIO_DIR = 'D:/Datasets/frog_calls/round_2'            # wav files (likely 16 kHz)
ANNOTATION_DIR = 'D:/Datasets/frog_calls/round_2'     # Raven .Table.1.selections.txt files

# Perch (v2)
PERCH_SAMPLE_RATE = 32000
PERCH_CLIP_SECONDS = 5.0
PERCH_CLIP_SAMPLES = int(PERCH_SAMPLE_RATE * PERCH_CLIP_SECONDS)  # 160000

# Dataset original audio sample rate (your files)
DATASET_SAMPLE_RATE = 16000  # as you stated; used for computing metadata ranges
CLIP_DURATION_SECONDS = PERCH_CLIP_SECONDS  # we use 5s windows for sampling

# Sampling / metadata
TEST_SPLIT = 0.05
POS_RATIO = 0.5
RANDOM_SEED = 42
METADATA_WORKERS = 8  # threads for metadata creation

# Training defaults
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
LABEL_MODE = 'count'  # 'binary' or 'count'

# Checkpoints
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
