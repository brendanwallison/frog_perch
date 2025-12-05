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

def train(
    label_mode=config.LABEL_MODE,
    epochs=config.EPOCHS,
    batch_size=config.BATCH_SIZE,
    pool_method=getattr(config, "POOL_METHOD", "conv"),
    val_stride=getattr(config, "VAL_STRIDE_SEC", 1.0),
    steps_per_epoch=getattr(config, "STEPS_PER_EPOCH", 200),
    q2_confidence=getattr(config, "Q2_CONFIDENCE", 0.75),
    equalize_q2_val=getattr(config, "EQUALIZE_Q2_VAL", False),
    use_continuous_confidence = getattr(config, "USE_CONTINUOUS_CONFIDENCE", False),
    confidence_params=getattr(config, "CONFIDENCE_PARAMS", None),
):
    """
    Train downstream model with:
      - random-crop training (infinite-like via generator)
      - deterministic stride-based validation (finite)
    Returns: (model, val_ds)
    """

    if confidence_params is None:
        confidence_params = {}


    # 1) instantiate dataset objects
    train_ds_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=True,
        pos_ratio=config.POS_RATIO,
        random_seed=config.RANDOM_SEED,
        label_mode=label_mode,
        q2_confidence=q2_confidence,
        use_continuous_confidence=use_continuous_confidence,
        confidence_params=confidence_params
    )

    val_ds_obj = FrogPerchDataset(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        train=False,
        pos_ratio=None,  # no rebalancing for validation
        random_seed=config.RANDOM_SEED,
        label_mode=label_mode,
        val_stride_sec=val_stride,
        q2_confidence=q2_confidence,
        equalize_q2_val=equalize_q2_val,
        use_continuous_confidence=use_continuous_confidence,
        confidence_params=confidence_params
    )

    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)

    # 2) build downstream model
    model = build_downstream(
        spatial_shape=(16, 4, 1536),
        label_mode=label_mode,
        pool_method=pool_method,
    )

    # 3) loss & metrics
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
        # loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        loss = SliceLossWithSoftCountKL(max_bin=4, kl_weight=1.0)
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="slice_acc"),
            tf.keras.metrics.AUC(name="slice_auc"),
            tf.keras.metrics.Precision(name="slice_precision"),
            tf.keras.metrics.Recall(name="slice_recall"),

            # Clip-level metrics derived from slice logits
            SliceToCountKLDivergence(),
            SliceExpectedCountMAE(),
            SliceEMD(),
            SliceExpectedRecall(),
            SliceExpectedPrecision(),
            SliceExpectedBinaryAccuracy(),
        ]

    else:
        # count mode: dataset returns a probability vector (soft counts)
        # use KLDivergence (you used this earlier) and CategoricalAccuracy
        loss = tf.keras.losses.KLDivergence()
        metrics = [
            ExpectedCountMAE(),
            EMD(),
            ExpectedRecall(),
            ExpectedPrecision(),
            ExpectedBinaryAccuracy()
        ]


    model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss=loss, metrics=metrics)

    # 4) callbacks
    q2_tag = int(round(q2_confidence * 100))
    logp = confidence_params.get("logistic_params", {})
    x0 = logp.get("x0", None)
    k  = logp.get("k", None)


    if x0 is not None and k is not None:
        fname = f"pool={pool_method}_loss={label_mode}_x0={x0}_k={k}.keras"
    else:
        fname = f"pool={pool_method}_loss={label_mode}_q2={q2_tag:03d}.keras"


    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, fname)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True),
        GPUMemoryCallback(),
    ]

    validation_steps = len(val_ds_obj) // batch_size

    # 5) fit
    model.fit(
        train_ds,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        verbose=2  # <-- ADD THIS LINE
    )

    return model, val_ds
