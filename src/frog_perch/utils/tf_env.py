import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

# Optional: eliminates oneDNN CPU warnings
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
