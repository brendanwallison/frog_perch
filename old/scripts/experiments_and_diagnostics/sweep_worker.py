#!/usr/bin/env python3
"""
Hardcoded worker that imports the evaluate helper by filepath (no package import).
Usage:
    python experiments/sweep_worker.py <x0> <k> <out_json>

This script will:
 - import tensorflow inside the process
 - run train.train(...) with modified confidence params
 - call the helper evaluate_without_smoothing (loaded by file path)
 - write atomic JSON to out_json (success or error)
 - exit
"""
import os
import sys
import json
import time
import traceback
import importlib.util
from pathlib import Path

# recommended env settings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_CUDNN_USE_AUTOTUNE", "0")
os.environ.setdefault("TF_ENABLE_XLA", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

if len(sys.argv) != 4:
    print("Usage: sweep_worker.py <x0> <k> <out_json>", file=sys.stderr)
    sys.exit(2)

x0 = float(sys.argv[1])
k  = float(sys.argv[2])
out_json = Path(sys.argv[3])

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER_PATH = REPO_ROOT / "experiments" / "logistic_sweep_helpers.py"

def atomic_save(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(str(tmp), str(path))

def load_helper_by_path(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"helper not found: {path}")
    spec = importlib.util.spec_from_file_location("logistic_sweep_helpers", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "evaluate_without_smoothing"):
        raise ImportError("helper missing evaluate_without_smoothing")
    return mod.evaluate_without_smoothing

start = time.time()
result = {"x0": x0, "k": k}

try:
    evaluate_without_smoothing = load_helper_by_path(HELPER_PATH)

    # import tensorflow and configure GPU growth
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

    # import repo training function & config
    import copy
    import frog_perch.config as config
    from frog_perch.training import train as train_module

    conf = copy.deepcopy(config.CONFIDENCE_PARAMS)
    conf["logistic_params"]["x0"] = x0
    conf["logistic_params"]["k"] = k

    model, val_ds = train_module.train(
        label_mode=config.LABEL_MODE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        pool_method=config.POOL_METHOD,
        q2_confidence=config.Q2_CONFIDENCE,
        equalize_q2_val=config.EQUALIZE_Q2_VAL,
        use_continuous_confidence=True,
        confidence_params=conf
    )

    hist = getattr(model, "history", None)
    best_val_loss = float(hist.history["val_loss"][-1]) if (hist and "val_loss" in hist.history and hist.history["val_loss"]) else None

    # evaluate via helper (it will save PR plots into same dir as out_json)
    try:
        ap_scores = evaluate_without_smoothing(model, config.BATCH_SIZE, f"x0={x0}_k={k}", str(out_json.parent))
    except Exception as e:
        # if helper fails, record but continue
        ap_scores = {"evaluate_error": str(e)}

    duration = time.time() - start
    result.update({
        "best_val_loss": best_val_loss,
        "seconds": float(duration),
        **{k: float(v) for k, v in ap_scores.items() if isinstance(v, (int, float))}
    })

    # atomic save
    atomic_save(out_json, result)

    # teardown TF
    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass
    try:
        import gc; gc.collect()
    except Exception:
        pass

    sys.exit(0)

except Exception as e:
    tb = traceback.format_exc()
    result["error"] = str(e)
    result["traceback"] = tb
    try:
        atomic_save(out_json, result)
    except Exception:
        try:
            with open(str(out_json) + ".err", "w") as f:
                f.write(str(e) + "\n\n" + tb)
        except Exception:
            pass
    sys.exit(1)
