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

# Sampling / metadata
TEST_SPLIT = 0.15
VAL_SPLIT = 0.1
POS_RATIO = 0.5
RANDOM_SEED = 41
METADATA_WORKERS = 8  # threads for metadata creation

# Training defaults
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH = 100

# Validation defaults
VAL_STRIDE_SEC = 1
EQUALIZE_Q2_VAL = False

# Checkpoints
CHECKPOINT_DIR = 'checkpoints/KL_04-17_transformer_l2_1e-3_td_64'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# NN Hyperparameters
SPATIAL_SHAPE = (16, 4, 1536)
SLICE_HIDDEN_DIMS = (512, 256)
TEMPORAL_DIM = 64
NUM_TEMPORAL_LAYERS = 5
KERNEL_SIZE = 3
ACTIVATION = 'gelu'
DROPOUT = 0.1
L2_REG = 0.001
USE_GATING = True
MAX_BIN = 16

# Continuous-confidence configuration
USE_CONTINUOUS_CONFIDENCE = True
CONFIDENCE_LOGISTIC_PARAMS = {
    'k': 1.0,
    'x0': -4.0,
    'lower': 0.0, 
    'upper': 1.0,
    'clip_z': 10.0
}

