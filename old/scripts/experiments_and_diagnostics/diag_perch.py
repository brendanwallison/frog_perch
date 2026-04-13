# diag_perch.py (WSL-aware)
import os, sys, platform
from pathlib import Path

# --- WSL / environment diagnostics ---
print("=== Environment Diagnostics ===")
print("Platform:", platform.platform())
print("System:", platform.system(), "| Release:", platform.release())
print("Python executable:", sys.executable)
print("PWD:", os.getcwd())

# Detect if running inside WSL
is_wsl = "microsoft" in platform.release().lower()
print("Running inside WSL:", is_wsl)

# Warn if running on Windows paths inside WSL (slow, breaks hardlinks)
cwd_path = Path(os.getcwd())
if str(cwd_path).startswith("/mnt/"):
    print("WARNING: You are inside a Windows-mounted filesystem (/mnt/*).")
    print("         This causes slow I/O, broken hardlinks, and TF issues.")
    print("         Move project to your WSL home: ~/dev/frog_perch")
print()

# --- Print TensorFlow-relevant env vars BEFORE importing TF ---
print("=== TF Environment Variables ===")
for var in [
    "TF_XLA_FLAGS",
    "XLA_FLAGS",
    "TF_ENABLE_ONEDNN_OPTS",
    "TF_CPP_MIN_LOG_LEVEL",
]:
    print(f"{var}: {os.environ.get(var)}")
print()

# --- Import TensorFlow ---
print("=== TensorFlow Diagnostics ===")
import tensorflow as tf

print("TF version:", tf.__version__)
print("Is built with CUDA:", tf.test.is_built_with_cuda())

# List all visible GPU devices
gpus = tf.config.list_physical_devices("GPU")
print("Visible GPU devices:", gpus)

# WSL-GPU specific GPU check
try:
    import subprocess
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if r.returncode == 0:
        print("nvidia-smi: GPU detected under WSL")
    else:
        print("nvidia-smi not available (no NVIDIA GPU or no WSLg CUDA support)")
except FileNotFoundError:
    print("nvidia-smi command not found (likely CPU-only WSL)")
print()

# --- Perch SavedModel path ---
print("=== Perch SavedModel Diagnostics ===")
from frog_perch.utils.download_perch import get_perch_savedmodel_path

sm = Path(get_perch_savedmodel_path())
print("Perch SavedModel directory:", sm)

if not sm.exists():
    print("ERROR: SavedModel directory does not exist!")
else:
    print("Directory contents:", [p.name for p in sm.iterdir()])

pb_files = list(sm.glob("**/saved_model.pb"))
print("Found saved_model.pb:", [str(p) for p in pb_files])
print()

# --- Attempt loading the SavedModel ---
print("=== Testing TF SavedModel Load ===")

# Disable JIT / MLIR for safety on WSL CPU-only
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.disable_mlir_bridge()
except Exception:
    pass

try:
    print("Loading SavedModel...")
    model = tf.saved_model.load(str(sm))
    print("Loaded successfully.")

    if hasattr(model, "signatures"):
        sigs = list(model.signatures.keys())
        print("Signatures:", sigs)

        if "serving_default" in model.signatures:
            sig = model.signatures["serving_default"]
            print("serving_default input signature:", sig.structured_input_signature)

            # Prepare dummy input if possible
            import numpy as np

            inputs = list(sig.structured_input_signature[1].items())
            if inputs:
                name, spec = inputs[0]
                print("Trying dummy input for:", name)

                try:
                    dummy = np.zeros((1, 160000), dtype=np.float32)
                    out = sig(**{name: tf.convert_to_tensor(dummy)})
                    print("serving_default call output keys:",
                          list(out.keys()) if isinstance(out, dict) else "tensor")
                except Exception as e:
                    print("Dummy call failed:", e)

except Exception as e:
    print("Failed to load Perch SavedModel:", repr(e))
    sys.exit(1)
