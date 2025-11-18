# frog_perch/utils/tf_env.py
import os

# Disable XLA device loading (Perch v2 otherwise loads a CUDA-only XLA binary)
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_enable_xla_devices=false")

# Ensure XLA still has *at least one* host device (fixes zero-thread crash)
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=1")

# Optional: eliminate oneDNN warnings and mismatches
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
