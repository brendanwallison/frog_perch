import os
import numpy as np
import tensorflow as tf

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.nn_models.downstream import build_downstream
from frog_perch.nn_training.dataset_builders import build_tf_dataset, build_tf_val_dataset

# Import custom metrics, losses, and callbacks
from frog_perch.nn_training.metrics import (
    AnnealedLossWrapper,
    NormalizedEarthMoversDistance1D,
    ExpectedCountMAE
)
from frog_perch.nn_training.callbacks import (
    GPUMemoryCallback,
    LossAnnealingCallback
)


def inspect_ds(ds_obj: FrogPerchDataset, name: str) -> None:
    expected_counts = []
    for i in range(len(ds_obj)):
        _, labels, _, _ = ds_obj[i]
        dist = np.array(labels["count_probs"])
        k = np.arange(len(dist))
        expected_counts.append(np.sum(k * dist))

    expected_counts = np.array(expected_counts)
    print(f"\n========== {name} STATS ==========")
    print(f"Samples: {len(expected_counts)}")
    print(f"Mean expected events: {expected_counts.mean():.4f}")
    print(f"Max expected events:  {expected_counts.max():.4f}")


def train(cfg: dict) -> tuple[tf.keras.Model, tf.data.Dataset]:
    """
    Trains the downstream frog perch model using a configuration dictionary.
    """
    # 1. Extract hyperparameters with sensible defaults
    epochs = cfg.get("EPOCHS", 100)
    batch_size = cfg.get("BATCH_SIZE", 32)
    val_stride = cfg.get("VAL_STRIDE_SEC", 1.0)
    steps_per_epoch = cfg.get("STEPS_PER_EPOCH", 100)
    confidence_params = cfg.get("CONFIDENCE_PARAMS", {})
    max_bin = cfg.get("MAX_BIN", 16)

    # 2. Setup Dataset arguments
    dataset_kwargs = dict(
        audio_dir=cfg["AUDIO_DIR"],             # Required
        annotation_dir=cfg["ANNOTATION_DIR"],   # Required
        random_seed=cfg.get("RANDOM_SEED", 42),
        confidence_params=confidence_params,
    )

    # 3. Build Datasets
    train_ds_obj = FrogPerchDataset(
        split_type='train', 
        pos_ratio=cfg.get("POS_RATIO", 0.5), 
        **dataset_kwargs
    )
    val_ds_obj   = FrogPerchDataset(split_type='val', val_stride_sec=val_stride, **dataset_kwargs)
    test_ds_obj  = FrogPerchDataset(split_type='test', val_stride_sec=val_stride, **dataset_kwargs)

    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds   = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)
    test_ds  = build_tf_val_dataset(test_ds_obj, batch_size=batch_size)

    inspect_ds(val_ds_obj, "VALIDATION")
    inspect_ds(test_ds_obj, "TEST")

    # 4. Build Model
    model = build_downstream(
        spatial_shape=cfg.get("SPATIAL_SHAPE", (16, 4, 1536)),
        slice_hidden_dims=cfg.get("SLICE_HIDDEN_DIMS", (512, 256)),
        temporal_dim=cfg.get("TEMPORAL_DIM", 256),
        num_temporal_layers=cfg.get("NUM_TEMPORAL_LAYERS", 2),
        kernel_size=cfg.get("KERNEL_SIZE", 3),
        activation=cfg.get("ACTIVATION", "gelu"),
        dropout=cfg.get("DROPOUT", 0.1),
        l2_reg=cfg.get("L2_REG", 1e-2),
        use_gating=cfg.get("USE_GATING", True),
        max_bin=max_bin
    )

    # 5. Define dynamic weights
    weight_binary = tf.Variable(0.1, trainable=False, dtype=tf.float32, name="weight_binary")
    weight_slice = tf.Variable(0.1, trainable=False, dtype=tf.float32, name="weight_slice")

    # Wrap the standard losses with the dynamic variables
    losses = {
        "binary": AnnealedLossWrapper(
            tf.keras.losses.BinaryCrossentropy(), 
            weight_binary, 
            name="annealed_binary"
        ),
        "slice": AnnealedLossWrapper(
            tf.keras.losses.BinaryCrossentropy(from_logits=True), 
            weight_slice, 
            name="annealed_slice"
        ),
        "count_probs": tf.keras.losses.KLDivergence(),
    }

    metrics = {
        "binary": [
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.AUC(name="bin_auc"),
        ],
        "slice": [
            tf.keras.metrics.BinaryAccuracy(name="slice_acc", threshold=0.0),
            tf.keras.metrics.AUC(name="slice_auc", from_logits=True), 
        ],
        "count_probs": [
            tf.keras.losses.KLDivergence(),
            ExpectedCountMAE(name="count_mae", max_bin=max_bin),
            NormalizedEarthMoversDistance1D()
        ],
    }

    # 6. Compile Model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.get("LEARNING_RATE", 1e-3)),
        loss=losses,
        metrics=metrics,
        #run_eagerly=True
    )

    # 7. Setup Callbacks
    checkpoint_dir = cfg.get("CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(checkpoint_dir, "best.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        ),
        GPUMemoryCallback(),
        LossAnnealingCallback(weight_binary, weight_slice, epochs)
    ]

    # 8. Train Model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        verbose=2,
        callbacks=callbacks,
    )

    print("\n[INFO] Evaluating on test set...")
    test_results = model.evaluate(test_ds)
    print(f"Test Results: {test_results}")

    return model, val_ds