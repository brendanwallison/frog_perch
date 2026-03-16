# src/frog_perch/training/train.py
import os
import numpy as np
import tensorflow as tf
import sys

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.models.downstream import build_downstream
import frog_perch.config as config

from frog_perch.training.dataset_builders import (
    build_tf_dataset,
    build_tf_val_dataset
)

from frog_perch.metrics.count_metrics import (
    ExpectedRecall,
    ExpectedPrecision,
    ExpectedBinaryAccuracy,
    EMD,
    ExpectedCountMAE
)

from frog_perch.metrics.slice_to_count_metrics import (
    SliceToCountWrapper,
    SliceToCountKLDivergence,
    SliceExpectedCountMAE,
    SliceEMD,
    SliceExpectedRecall,
    SliceExpectedPrecision,
    SliceExpectedBinaryAccuracy,
    SliceLossWithSoftCountKL
)

class GPUMemoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            info = tf.config.experimental.get_memory_info('GPU:0')
            used = info['current'] / (1024 ** 2)
            peak = info['peak'] / (1024 ** 2)
            
            # Use \r to move to the beginning of the line
            # Use no \n, and add padding spaces to ensure old text is overwritten
            output = (
                f"\r[GPU MEMORY] Epoch {epoch+1}: used={used:.0f} MB | peak={peak:.0f} MB   " 
            )
            sys.stdout.write(output)
            sys.stdout.flush()
            
        except Exception:
            pass # ...

def inspect_ds(ds_obj, name):
    all_labels = []
    
    # Iterate through the custom dataset object (which yields 4 items)
    for i in range(len(ds_obj)):
        # Unpack the 4 values your dataset_builders.py expects
        spatial, label, audio_file, start = ds_obj[i]
        
        # Convert to numpy and handle potential scalars
        label_val = np.array(label)
        all_labels.append(label_val)
    
    all_labels = np.array(all_labels)
    
    # Calculate stats
    mean_val = np.mean(all_labels)
    max_val = np.max(all_labels)
    zeros_ratio = np.mean(all_labels == 0)

    print(f"\n{'='*10} {name} STATS {'='*10}")
    print(f"Total Samples: {len(all_labels)}")
    print(f"Label Mean:    {mean_val:.4f}  (Higher = more frogs/higher counts)")
    print(f"Label Max:     {max_val:.4f}")
    print(f"Sparsity:      {zeros_ratio*100:.1f}% are zero-labels")
    
    # For counting/slice tasks, distribution matters
    if all_labels.ndim > 1:
        # If labels are soft-counts/distributions, check the sum
        avg_sum = np.mean(np.sum(all_labels, axis=-1))
        print(f"Avg Count Sum: {avg_sum:.4f}")


def train(
    label_mode=config.LABEL_MODE,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    pool_method=getattr(config, "POOL_METHOD", "conv"),
    val_stride=getattr(config, "VAL_STRIDE_SEC", 1.0),
    steps_per_epoch=getattr(config, "STEPS_PER_EPOCH", 100),
    q2_confidence=getattr(config, "Q2_CONFIDENCE", 0.75),
    equalize_q2_val=getattr(config, "EQUALIZE_Q2_VAL", False),
    use_continuous_confidence = getattr(config, "USE_CONTINUOUS_CONFIDENCE", False),
    confidence_params=getattr(config, "CONFIDENCE_PARAMS", None),
):
    """
    Train downstream model with:
      - 3-way explicit split (train, val, test)
      - Logic to resume from existing .keras checkpoints
      - Random-crop training & deterministic stride-based validation/test
    """

    if confidence_params is None:
        confidence_params = {}

    # 1) Instantiate dataset objects for all three splits
    dataset_kwargs = dict(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        random_seed=config.RANDOM_SEED,
        label_mode=label_mode,
        q2_confidence=q2_confidence,
        use_continuous_confidence=use_continuous_confidence,
        confidence_params=confidence_params
    )

    train_ds_obj = FrogPerchDataset(split_type='train', pos_ratio=config.POS_RATIO, **dataset_kwargs)
    val_ds_obj   = FrogPerchDataset(split_type='val', val_stride_sec=val_stride, equalize_q2_val=equalize_q2_val, **dataset_kwargs)
    test_ds_obj  = FrogPerchDataset(split_type='test', val_stride_sec=val_stride, equalize_q2_val=equalize_q2_val, **dataset_kwargs)
    # test_ds_obj   = FrogPerchDataset(split_type='val', val_stride_sec=val_stride, equalize_q2_val=equalize_q2_val, **dataset_kwargs)
    # val_ds_obj  = FrogPerchDataset(split_type='test', val_stride_sec=val_stride, equalize_q2_val=equalize_q2_val, **dataset_kwargs)


    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds   = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)
    test_ds  = build_tf_val_dataset(test_ds_obj, batch_size=batch_size)

    inspect_ds(val_ds_obj, "VALIDATION")
    inspect_ds(test_ds_obj, "TEST")

    # 2) Define Checkpoint Path (keeping your naming logic)
    q2_tag = int(round(q2_confidence * 100))
    logp = confidence_params.get("logistic_params", {})
    x0, k = logp.get("x0"), logp.get("k")

    if x0 is not None and k is not None:
        fname = f"pool={pool_method}_loss={label_mode}_x0={x0}_k={k}.keras"
    else:
        fname = f"pool={pool_method}_loss={label_mode}_q2={q2_tag:03d}.keras"

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, fname)

    # 3) Setup custom objects for loading
    custom_objects = {
        "SliceToCountWrapper": SliceToCountWrapper,
        "SliceLossWithSoftCountKL": SliceLossWithSoftCountKL,
        "SliceToCountKLDivergence": SliceToCountKLDivergence,
        "SliceExpectedCountMAE": SliceExpectedCountMAE,
        "SliceEMD": SliceEMD,
        "SliceExpectedRecall": SliceExpectedRecall,
        "SliceExpectedPrecision": SliceExpectedPrecision,
        "SliceExpectedBinaryAccuracy": SliceExpectedBinaryAccuracy,
        "ExpectedCountMAE": ExpectedCountMAE,
        "EMD": EMD,
        "ExpectedRecall": ExpectedRecall,
        "ExpectedPrecision": ExpectedPrecision,
        "ExpectedBinaryAccuracy": ExpectedBinaryAccuracy,
    }

    # 4) Build or Load Model
    if os.path.exists(ckpt_path):
        print(f"\n[INFO] Found checkpoint at {ckpt_path}. Resuming training...")
        model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)
    else:
        print(f"\n[INFO] No checkpoint found at {ckpt_path}. Building new model...")
        model = build_downstream(
            spatial_shape=(16, 4, 1536),
            label_mode=label_mode,
            pool_method=pool_method,
        )

        # Define loss and metrics for fresh compilation
        if label_mode == 'binary':
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="acc"),
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.MeanSquaredError(name="brier"),
            ]
        elif label_mode == 'slice':
            loss = SliceLossWithSoftCountKL(max_bin=16, kl_weight=2.0)
            metrics = [
                tf.keras.metrics.BinaryAccuracy(name="slice_acc"),
                tf.keras.metrics.AUC(name="slice_auc"),
                tf.keras.metrics.Precision(name="slice_precision"),
                tf.keras.metrics.Recall(name="slice_recall"),
                SliceToCountKLDivergence(),
                SliceExpectedCountMAE(),
                SliceEMD(),
                SliceExpectedRecall(),
                SliceExpectedPrecision(),
                SliceExpectedBinaryAccuracy(),
            ]
        else:
            loss = tf.keras.losses.KLDivergence()
            metrics = [
                ExpectedCountMAE(), EMD(), ExpectedRecall(),
                ExpectedPrecision(), ExpectedBinaryAccuracy()
            ]

        model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss=loss, metrics=metrics)

    # 5) Callbacks and Fit
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        GPUMemoryCallback(),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=len(val_ds_obj) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=2
    )

    # 6) Final Evaluation on Test Set
    print("\n[INFO] Training complete. Evaluating on hold-out test set...")
    test_results = model.evaluate(test_ds, steps=len(test_ds_obj) // batch_size)
    print(f"Test Results: {test_results}")

    # Manually evaluate the validation set using the .evaluate() method
    val_check = model.evaluate(val_ds, steps=len(val_ds_obj) // batch_size)
    print(f"Fit Val Loss: {history.history['val_loss'][-1]}")
    print(f"Manual Val Loss: {val_check[0]}")

    return model, val_ds