# config.py
import os

# Paths: set these to your local environment
AUDIO_DIR = '/home/breallis/datasets/frog_calls/round_2'            # wav files (likely 16 kHz)
ANNOTATION_DIR = '/home/breallis/datasets/frog_calls/round_2'     # Raven .Table.1.selections.txt files

# Perch (v2)
PERCH_SAMPLE_RATE = 32000
PERCH_CLIP_SECONDS = 5.0
PERCH_CLIP_SAMPLES = int(PERCH_SAMPLE_RATE * PERCH_CLIP_SECONDS)  # 160000

# Dataset original audio sample rate (your files)
DATASET_SAMPLE_RATE = 16000  # as you stated; used for computing metadata ranges
CLIP_DURATION_SECONDS = PERCH_CLIP_SECONDS  # we use 5s windows for sampling
Q2_CONFIDENCE = 1.0

# Sampling / metadata
TEST_SPLIT = 0.3
POS_RATIO = 0.5
RANDOM_SEED = 42
METADATA_WORKERS = 8  # threads for metadata creation

# Training defaults
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 1e-5
LABEL_MODE = 'count'  # 'binary' or 'count'
STEPS_PER_EPOCH = 250

# Validation defaults
VAL_STRIDE_SEC = 1
EQUALIZE_Q2_VAL = True

# Checkpoints
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Downstream model pooling method
# Options: 'mean','conv','avgmax','mlp_flat','attn','conv2','bottleneck1x1','temporal','freq'
POOL_METHOD = 'mlp_flat'


# ============================================================
# Continuous-confidence configuration
# ============================================================

USE_CONTINUOUS_CONFIDENCE = True

CONFIDENCE_PARAMS = {
    "duration_stats": {
        "mean": 0.09197873701622324,
        "std": 0.019943076189245984,
    },
    "bandwidth_stats": {
        "mean": 1019.8970794938352,
        "std": 246.82156166917892,
    },
    "logistic_params": {
        # These are example defaults — adjust based on tuning
        'k': 1.0,
        'x0': -4.0,
        'lower': 0.0, 
        'upper': 1.0,
        'clip_z': 10.0
    }
}
