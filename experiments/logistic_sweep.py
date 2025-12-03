#!/usr/bin/env python3
import os

# Silence TensorFlow/XLA C++ backend logs (INFO, WARNING, ERROR)
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["TF_DETERMINISTIC_OPS"] = "1"
# os.environ["TF_CUDNN_USE_AUTOTUNE"] = "0"
# os.environ["TF_ENABLE_XLA"] = "0"


import json
import csv
import itertools
import time
import copy
import subprocess
import sys
from pathlib import Path

# Light imports only
import frog_perch.config as config

# ================================================================
# Helpers
# ================================================================
def configure_gpu():
    """Configure GPU memory growth for clean exit."""
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except Exception:
            pass

def train_with_smoothing(x0, k):
    """Run training with label smoothing enabled for sweep."""
    from frog_perch.training import train
    conf = copy.deepcopy(config.CONFIDENCE_PARAMS)
    conf["logistic_params"]["x0"] = x0
    conf["logistic_params"]["k"] = k

    model, val_ds = train.train(
        label_mode=config.LABEL_MODE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        pool_method=config.POOL_METHOD,
        q2_confidence=config.Q2_CONFIDENCE,
        equalize_q2_val=config.EQUALIZE_Q2_VAL,
        use_continuous_confidence=True,
        confidence_params=conf
    )
    return model, val_ds

def evaluate_without_smoothing(model, batch_size, tag, sweep_dir):
    """
    Ultimate validation: rebuild val_ds with no label smoothing and compute
    one-vs-rest precision-recall curves for each class.
    """
    from frog_perch.datasets.frog_dataset import FrogPerchDataset
    from frog_perch.training.dataset_builders import build_tf_val_dataset
    from sklearn.metrics import precision_recall_curve, average_precision_score
    import matplotlib
    matplotlib.use("Agg")  # safe for headless servers
    import matplotlib.pyplot as plt
    import numpy as np

    # Build validation dataset with hard labels
    val_ds_obj_final = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False,
        pos_ratio=None,
        random_seed=config.RANDOM_SEED,
        label_mode=config.LABEL_MODE,
        val_stride_sec=1.0,
        q2_confidence=config.Q2_CONFIDENCE,
        equalize_q2_val=True,
        use_continuous_confidence=False,
        confidence_params={}
    )
    val_ds_final = build_tf_val_dataset(val_ds_obj_final, batch_size=batch_size)

    # Collect predictions and true labels
    y_true = np.concatenate([y for _, y in val_ds_final], axis=0)
    y_pred = model.predict(val_ds_final)

    n_classes = y_true.shape[1]
    ap_scores = {}

    # One-vs-rest PR curve for each class
    for i in range(n_classes):
        y_true_bin = y_true[:, i].astype(int)
        y_pred_bin = y_pred[:, i]

        prec, rec, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        ap = average_precision_score(y_true_bin, y_pred_bin)
        ap_scores[f"class_{i}_AP"] = float(ap)

        # Save PR curve plot
        plt.figure()
        plt.plot(rec, prec, label=f"Class {i} (AP={ap:.3f})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve — {tag}, Class {i}")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(sweep_dir, f"{tag}_pr_class{i}.png"))
        plt.close()

    return ap_scores

def save_result(result_path, result):
    """Save JSON result safely to disk."""
    Path(os.path.dirname(result_path)).mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
        f.flush(); os.fsync(f.fileno())


# ================================================================
# Worker function
# ================================================================
def run_single_sweep(x0, k, sweep_dir):
    tag = f"x0={x0}_k={k}"
    result_path = os.path.join(sweep_dir, f"{tag}.json")

    if os.path.exists(result_path):
        print(f"Skipping {tag} (already exists).", flush=True)
        os._exit(0)

    # Import TF inside worker
    import tensorflow as tf
    configure_gpu()

    print(f"\n=== Running sweep: {tag} ===", flush=True)
    start = time.time()

    # Train with smoothing
    model, val_ds = train_with_smoothing(x0, k)
    duration = time.time() - start
    best_val_loss = float(model.history.history["val_loss"][-1])

    # Ultimate validation without smoothing
    ap_scores = evaluate_without_smoothing(model, config.BATCH_SIZE, tag, sweep_dir)

    # Save JSON result
    result = {
        "x0": x0,
        "k": k,
        "best_val_loss": best_val_loss,
        "seconds": duration,
    }
    result.update(ap_scores)

    save_result(result_path, result)

    # Cleanup
    try: tf.keras.backend.clear_session()
    except Exception: pass
    try:
        import gc; gc.collect()
    except Exception: pass

    print(f"Completed {tag}: val_loss={best_val_loss:.4f}, AP scores={ap_scores}", flush=True)

    os._exit(0)


# ================================================================
# Main entry point — controller
# ================================================================
def main():
    # Output directory must be created here
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sweep_dir = os.path.join(script_dir, "logistic_sweep_results")
    os.makedirs(sweep_dir, exist_ok=True)

    # Parameter ranges
    # x0_values = [-2, -1, -0.5, 0, 0.5, 1, 2]
    # k_values  = [0.25, 0.5, 1.0, 2.0, 4.0]

    x0_values = [-5, -3]
    k_values  = [0.5, 1.0]

    sweep_items = list(itertools.product(x0_values, k_values))

    print(f"\nLaunching sweep with {len(sweep_items)} configs...\n")

    # Strict sequential execution — only one GPU user at a time
    for x0, k in sweep_items:
        tag = f"x0={x0}_k={k}"
        result_path = os.path.join(sweep_dir, f"{tag}.json")
        print(f"Starting subprocess for {tag}")

        # Skip if already done
        if os.path.exists(result_path):
            print(f"Already completed: {tag}")
            continue

        # Launch fresh interpreter with unbuffered mode (-u) to stream output
        cmd = [
            sys.executable,
            "-u",           # unbuffer stdout/stderr
            __file__,
            "--worker",
            str(x0),
            str(k),
            sweep_dir,
        ]

        # Stream child output live
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        ) as proc:
            # Read stdout and stderr concurrently in a simple loop
            # (interleave lines for visibility; avoid blocking)
            while True:
                out_line = proc.stdout.readline()
                err_line = proc.stderr.readline()

                if out_line:
                    print(out_line, end="")  # already newline; preserve
                if err_line:
                    # Tag stderr so you can distinguish warnings/errors
                    print(f"[child:stderr] {err_line}", end="")

                if proc.poll() is not None:
                    # Drain any remaining output
                    remaining_out = proc.stdout.read()
                    remaining_err = proc.stderr.read()
                    if remaining_out:
                        print(remaining_out, end="")
                    if remaining_err:
                        print(f"[child:stderr] {remaining_err}", end="")
                    break

            rc = proc.returncode

        # Check completion artifact and return code
        if rc != 0:
            print(f"WARNING: subprocess for {tag} exited with code {rc}")
        if not os.path.exists(result_path):
            print(f"WARNING: no JSON result for {tag}. Likely stalled or crashed before save.")

        # Optional: small pause to let GPU driver settle between runs
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL)
        time.sleep(5.0)

    # Build summary CSV
    results = []
    for x0, k in sweep_items:
        tag = f"x0={x0}_k={k}"
        result_path = os.path.join(sweep_dir, f"{tag}.json")
        if os.path.exists(result_path):
            with open(result_path, "r") as f:
                results.append(json.load(f))

    if results:
        # Dynamically collect all keys across results
        fieldnames = set()
        for row in results:
            fieldnames.update(row.keys())
        # Ensure consistent ordering: put core fields first
        core_fields = ["x0", "k", "best_val_loss", "seconds"]
        # Then any per-class AP scores (class_0_AP, class_1_AP, ...)
        extra_fields = sorted([f for f in fieldnames if f not in core_fields])
        fieldnames = core_fields + extra_fields

        csv_path = os.path.join(sweep_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        print("\nSweep complete.")
        print(f"Summary written to: {csv_path}")
    else:
        print("No results found; summary CSV not created.")


# ================================================================
# REQUIRED: Only call main() when run as script
# ================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        # Worker mode
        x0 = float(sys.argv[2])
        k = float(sys.argv[3])
        sweep_dir = sys.argv[4]
        run_single_sweep(x0, k, sweep_dir)
    else:
        # Controller mode
        main()