# diag_perch.py
import os, sys
# Must set before importing tf — but here we want to print current env too
print("ENV TF_XLA_FLAGS:", os.environ.get("TF_XLA_FLAGS"))
print("ENV XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("ENV TF_ENABLE_ONEDNN_OPTS:", os.environ.get("TF_ENABLE_ONEDNN_OPTS"))
print("ENV TF_CPP_MIN_LOG_LEVEL:", os.environ.get("TF_CPP_MIN_LOG_LEVEL"))

# Import tensorflow after printing env
import tensorflow as tf
from pathlib import Path
from frog_perch.utils.download_perch import get_perch_savedmodel_path

print("TF version:", tf.__version__)
print("tf.test.is_built_with_cuda():", tf.test.is_built_with_cuda())
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

# Where is the cached Perch savedmodel?
sm = Path(get_perch_savedmodel_path())
print("Perch SavedModel directory:", sm)
print("Contains:", [p.name for p in sm.iterdir()])

# Try to list saved_model.pb location(s)
pb_files = list(sm.glob("**/saved_model.pb"))
print("saved_model.pb instances:", [str(p) for p in pb_files])

# Try to load model (wrapped try/except)
try:
    # Try disabling jit at runtime (best-effort)
    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    try:
        tf.config.experimental.disable_mlir_bridge()
    except Exception:
        pass

    print("Attempting to load SavedModel (this may error)...")
    model = tf.saved_model.load(str(sm))
    print("Loaded OK. signatures:", list(model.signatures.keys()) if hasattr(model, "signatures") else "no signatures")
    # If signatures exist, call serving_default with dummy input shape if possible
    if hasattr(model, "signatures") and "serving_default" in model.signatures:
        sig = model.signatures["serving_default"]
        print("serving_default inputs:", sig.structured_input_signature)
        # Prepare a zero waveform of expected length if possible (don't crash)
        # Try to find a numeric input shape
        import numpy as np
        # Heuristic: take first input and try shape e.g. [1,160000]
        inputs = list(sig.structured_input_signature[1].items())
        if inputs:
            name, spec = inputs[0]
            print("First input name:", name, "spec:", spec)
            # Attempt to create zeros for common shapes
            try:
                dummy = np.zeros((1, 160000), dtype=np.float32)
                out = sig(**{name: tf.convert_to_tensor(dummy)})
                print("serving_default call keys:", list(out.keys()) if isinstance(out, dict) else "tensor")
            except Exception as e:
                print("Calling serving_default failed:", e)
except Exception as e:
    print("Failed to load Perch SavedModel:", repr(e))
    sys.exit(1)
