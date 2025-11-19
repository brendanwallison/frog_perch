# dataset_builders.py  (drop-in replacement)
import tensorflow as tf
import numpy as np

# -------------------------
# Helpers
# -------------------------
def _label_to_rank1(label):
    """
    Convert scalar labels -> rank-1 np.array([label], dtype=float32)
    Convert vector labels -> np.asarray(..., dtype=float32)
    """
    if np.isscalar(label):
        return np.array([label], dtype=np.float32)
    return np.asarray(label, dtype=np.float32)


def _probe_sample(dataset_obj):
    """
    Return (spatial_sample, label_sample) from dataset_obj[0]
    """
    spatial_sample, label_sample, _, _ = dataset_obj[0]
    return spatial_sample, label_sample


# -------------------------
# Generators
# -------------------------
def _train_python_gen(dataset_obj):
    """
    Infinite generator for training. Yields:
      (spatial.astype(np.float32),
       label_out (rank-1 np.float32),
       np.bytes_(audio_file),
       np.int32(start))
    """
    length = len(dataset_obj)
    while True:
        spatial, label, audio_file, start = dataset_obj[np.random.randint(0, length)]
        label_out = _label_to_rank1(label)
        yield (
            spatial.astype(np.float32),
            label_out,
            np.bytes_(audio_file),
            np.int32(start),
        )


def _val_python_gen(dataset_obj):
    """
    Finite generator for validation. Yields same shapes/types as train generator.
    """
    for i in range(len(dataset_obj)):
        spatial, label, audio_file, start = dataset_obj[i]
        label_out = _label_to_rank1(label)
        yield (
            spatial.astype(np.float32),
            label_out,
            np.bytes_(audio_file),
            np.int32(start),
        )


# -------------------------
# Public builders
# -------------------------
def build_tf_dataset(dataset_obj, batch_size):
    """
    Build an infinite streaming tf.data.Dataset for training.
    The label spec is inferred as rank-1 (scalar -> [1], vector -> [K]).
    """
    # Probe one item for shapes
    spatial_sample, label_sample = _probe_sample(dataset_obj)

    spatial_spec = tf.TensorSpec(shape=tuple(spatial_sample.shape), dtype=tf.float32)

    # Ensure label spec is rank-1 by using _label_to_rank1
    label_arr = _label_to_rank1(label_sample)
    label_spec = tf.TensorSpec(shape=tuple(label_arr.shape), dtype=tf.float32)

    output_signature = (
        spatial_spec,
        label_spec,
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = (
        tf.data.Dataset
        .from_generator(lambda: _train_python_gen(dataset_obj), output_signature=output_signature)
        .repeat()
        .map(lambda s, l, a, b: (s, l), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


def build_tf_val_dataset(dataset_obj, batch_size):
    """
    Build a finite validation tf.data.Dataset.
    Signature is inferred identically to build_tf_dataset so both match.
    """
    if len(dataset_obj) == 0:
        raise RuntimeError("Validation dataset has zero length")

    # Probe first element
    first_spatial, first_label = _probe_sample(dataset_obj)

    spatial_spec = tf.TensorSpec(shape=tuple(first_spatial.shape), dtype=tf.float32)

    label_arr = _label_to_rank1(first_label)
    label_spec = tf.TensorSpec(shape=tuple(label_arr.shape), dtype=tf.float32)

    output_signature = (
        spatial_spec,
        label_spec,
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _val_python_gen(dataset_obj),
        output_signature=output_signature
    )

    ds = ds.map(lambda s, l, a, b: (s, l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
