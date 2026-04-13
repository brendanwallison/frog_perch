import os
import sys
import numpy as np
import tensorflow as tf

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.nn_models.downstream import build_downstream
import frog_perch.config as config
from frog_perch.nn_training.dataset_builders import build_tf_dataset, build_tf_val_dataset


class GPUMemoryCallback(tf.keras.callbacks.Callback):
    """Logs GPU memory usage at the end of each epoch."""
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        try:
            info = tf.config.experimental.get_memory_info('GPU:0')
            used = info['current'] / (1024 ** 2)
            peak = info['peak'] / (1024 ** 2)
            sys.stdout.write(f"\r[GPU MEMORY] Epoch {epoch+1}: used={used:.0f} MB | peak={peak:.0f} MB   ")
            sys.stdout.flush()
        except Exception:
            pass


class LossAnnealingCallback(tf.keras.callbacks.Callback):
    """Linearly decays binary and slice loss weights over the course of training."""
    
    def __init__(self, weight_binary: tf.Variable, weight_slice: tf.Variable, total_epochs: int):
        super().__init__()
        self.weight_binary = weight_binary
        self.weight_slice = weight_slice
        self.total_epochs = float(total_epochs)
        
        self.start_binary = 0.1
        self.end_binary = 0.0
        
        self.start_slice = 1.0
        self.end_slice = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        progress = epoch / self.total_epochs
        
        new_bin = self.start_binary - progress * (self.start_binary - self.end_binary)
        new_slice = self.start_slice - progress * (self.start_slice - self.end_slice)
        
        self.weight_binary.assign(max(new_bin, self.end_binary))
        self.weight_slice.assign(max(new_slice, self.end_slice))
        
        print(f"\n[ANNEAL] binary_weight: {new_bin:.4f} | slice_weight: {new_slice:.4f}")

@tf.keras.utils.register_keras_serializable(package="frog_perch")
class CountKLDivergence(tf.keras.metrics.Metric):
    """
    Stable KL divergence metric for distribution outputs.
    Always executes under both eager + graph mode.
    """

    def __init__(self, name="count_kl", **kwargs):
        super().__init__(name=name, **kwargs)

        self.kl_sum = self.add_weight(name="kl_sum", initializer="zeros")
        self.count_probs = self.add_weight(name="count_probs", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Normalize defensively (prevents silent NaN / skip behavior)
        y_true = y_true / tf.reduce_sum(y_true, axis=-1, keepdims=True)
        y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)

        kl = tf.reduce_sum(
            y_true * tf.math.log(tf.maximum(y_true, 1e-8) / tf.maximum(y_pred, 1e-8)),
            axis=-1,
        )

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            kl = kl * sample_weight
            self.count_probs.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count_probs.assign_add(tf.cast(tf.shape(kl)[0], tf.float32))

        self.kl_sum.assign_add(tf.reduce_sum(kl))

    def result(self):
        return tf.math.divide_no_nan(self.kl_sum, self.count_probs)

    def reset_state(self):
        self.kl_sum.assign(0.0)
        self.count_probs.assign(0.0)

@tf.keras.utils.register_keras_serializable(package="frog_perch")
class AnnealedLossWrapper(tf.keras.losses.Loss):
    """Dynamically applies a tf.Variable weight to a base loss during graph execution."""
    def __init__(self, base_loss, weight_var, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_loss = base_loss
        self.weight_var = weight_var

    def call(self, y_true, y_pred):
        # The multiplication happens inside the traced graph, guaranteeing dynamic updates
        return self.base_loss(y_true, y_pred) * self.weight_var


@tf.keras.utils.register_keras_serializable(package="frog_perch")
class NormalizedEarthMoversDistance1D(tf.keras.losses.Loss):
    """Computes the normalized 1D EMD between ordinal distributions."""
    
    def __init__(self, name="emd_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        cdf_true = tf.cumsum(y_true, axis=-1)
        cdf_pred = tf.cumsum(y_pred, axis=-1)
        
        # reduce_mean decouples gradient magnitude from max_bin sizing
        return tf.reduce_mean(tf.abs(cdf_true - cdf_pred), axis=-1)


@tf.keras.utils.register_keras_serializable(package="frog_perch")
class ExpectedCountMAE(tf.keras.metrics.Metric):
    """Computes the Mean Absolute Error between the expected counts of two distributions."""
    
    def __init__(self, max_bin=16, name="count_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_bin = max_bin
        self.total_mae = self.add_weight(name="total_mae", initializer="zeros")
        self.count_probs = self.add_weight(name="count_probs", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # FIX: Use static range instead of dynamic tf.shape to survive Keras graph tracing
        bins = tf.range(self.max_bin + 1, dtype=tf.float32)

        expected_true = tf.reduce_sum(bins * y_true, axis=-1)
        expected_pred = tf.reduce_sum(bins * y_pred, axis=-1)

        mae = tf.abs(expected_true - expected_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            mae = tf.multiply(mae, sample_weight)
            self.count_probs.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count_probs.assign_add(tf.cast(tf.shape(mae)[0], tf.float32))

        self.total_mae.assign_add(tf.reduce_sum(mae))

    def result(self):
        return tf.math.divide_no_nan(self.total_mae, self.count_probs)

    def reset_state(self):
        self.total_mae.assign(0.0)
        self.count_probs.assign(0.0)


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


def train(
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    val_stride: float = getattr(config, "VAL_STRIDE_SEC", 1.0),
    steps_per_epoch: int = getattr(config, "STEPS_PER_EPOCH", 100),
    confidence_params: dict = getattr(config, "CONFIDENCE_PARAMS", None),
) -> tuple[tf.keras.Model, tf.data.Dataset]:
    if confidence_params is None:
        confidence_params = {}

    dataset_kwargs = dict(
        audio_dir=config.AUDIO_DIR,
        annotation_dir=config.ANNOTATION_DIR,
        random_seed=config.RANDOM_SEED,
        confidence_params=confidence_params,
    )

    train_ds_obj = FrogPerchDataset(split_type='train', pos_ratio=config.POS_RATIO, **dataset_kwargs)
    val_ds_obj   = FrogPerchDataset(split_type='val', val_stride_sec=val_stride, **dataset_kwargs)
    test_ds_obj  = FrogPerchDataset(split_type='test', val_stride_sec=val_stride, **dataset_kwargs)

    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds   = build_tf_val_dataset(val_ds_obj, batch_size=batch_size)
    test_ds  = build_tf_val_dataset(test_ds_obj, batch_size=batch_size)

    inspect_ds(val_ds_obj, "VALIDATION")
    inspect_ds(test_ds_obj, "TEST")

    model = build_downstream(
        spatial_shape=getattr(config, "SPATIAL_SHAPE", (16, 4, 1536)),
        slice_hidden_dims=getattr(config, "SLICE_HIDDEN_DIMS", (512, 256)),
        temporal_dim=getattr(config, "TEMPORAL_DIM", 256),
        num_temporal_layers=getattr(config, "NUM_TEMPORAL_LAYERS", 2),
        kernel_size=getattr(config, "KERNEL_SIZE", 3),
        dropout=getattr(config, "DROPOUT", 0.1),
        l2_reg=getattr(config, "L2_REG", 1e-2),
        use_gating=getattr(config, "USE_GATING", True),
        max_bin=16
    )

    # Define dynamic weights
    weight_binary = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="weight_binary")
    weight_slice = tf.Variable(1.0, trainable=False, dtype=tf.float32, name="weight_slice")

    # Wrap the standard losses with the dynamic variables
    losses = {
        "binary": AnnealedLossWrapper(
            tf.keras.losses.BinaryCrossentropy(), 
            weight_binary, 
            name="annealed_binary"
        ),
        "slice_logits": AnnealedLossWrapper(
            tf.keras.losses.BinaryCrossentropy(from_logits=True), 
            weight_slice, 
            name="annealed_slice"
        ),
        # "count_probs": NormalizedEarthMoversDistance1D(),
        "count_probs": tf.keras.losses.KLDivergence(),
    }

    metrics = {
        "binary": [
            tf.keras.metrics.BinaryAccuracy(name="bin_acc"),
            tf.keras.metrics.AUC(name="bin_auc"),
        ],
        "slice_logits": [
            tf.keras.metrics.BinaryAccuracy(name="slice_acc", threshold=0.0),
            # FIX: Use the native from_logits argument
            tf.keras.metrics.AUC(name="slice_auc", from_logits=True), 
        ],
        "count_probs": [
            tf.keras.losses.KLDivergence(),
            ExpectedCountMAE(name="count_mae", max_bin=16),
            NormalizedEarthMoversDistance1D()
        ],
    }

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        loss=losses,
        metrics=metrics,
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.CHECKPOINT_DIR, "best.keras"),
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