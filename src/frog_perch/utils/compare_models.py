# compare_models.py
import frog_perch.config as config
from frog_perch.training.train import train
import tensorflow as tf
import json
import os
from datetime import datetime

POOL_METHODS = [
    'bottleneck1x1',
    'attn',
    'conv',
    'mean',
    'mlp_flat',
    'conv2'
]

def run_experiment(pool_method):
    print(f"\n===== TRAINING MODEL: {pool_method} =====")

    model, val_ds = train(
        label_mode=config.LABEL_MODE,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        pool_method=pool_method
    )

    val_loss, *metrics = model.evaluate(val_ds, verbose=1)

    return {
        "pool_method": pool_method,
        "val_loss": float(val_loss),
        "metrics": [float(m) for m in metrics]
    }

if __name__ == "__main__":
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"model_comparison_{timestamp}.json"

    # Run each model one after another
    for method in POOL_METHODS:
        # reset TF state between runs
        tf.keras.backend.clear_session()
        results.append(run_experiment(method))

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll experiments complete. Saved to {save_path}")
