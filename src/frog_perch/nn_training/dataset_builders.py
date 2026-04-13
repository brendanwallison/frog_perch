import tensorflow as tf
import numpy as np


# Helpers
def _probe_sample(dataset_obj):
    """
    Returns:
        (spatial_sample, label_dict_sample)
    """
    spatial_sample, label_sample, _, _ = dataset_obj[0]
    return spatial_sample, label_sample


def _train_python_gen(dataset_obj):
    """
    Infinite training generator.

    Yields:
        spatial (float32),
        labels dict (binary, count_probs, slice),
        audio_file (bytes),
        start (int32)
    """
    length = len(dataset_obj)

    while True:
        spatial, labels, audio_file, start = dataset_obj[
            np.random.randint(0, length)
        ]

        yield (
            spatial.astype(np.float32),
            {
                "binary": np.asarray(labels["binary"], dtype=np.float32),
                "count_probs": np.asarray(labels["count_probs"], dtype=np.float32),
                "slice": np.asarray(labels["slice"], dtype=np.float32),
            },
            np.bytes_(audio_file),
            np.int32(start),
        )


def _val_python_gen(dataset_obj):
    """
    Finite validation generator.
    """
    for i in range(len(dataset_obj)):
        spatial, labels, audio_file, start = dataset_obj[i]

        yield (
            spatial.astype(np.float32),
            {
                "binary": np.asarray(labels["binary"], dtype=np.float32),
                "count_probs": np.asarray(labels["count_probs"], dtype=np.float32),
                "slice": np.asarray(labels["slice"], dtype=np.float32),
            },
            np.bytes_(audio_file),
            np.int32(start),
        )


# Builders
def build_tf_dataset(dataset_obj, batch_size):
    """
    Build an infinite streaming tf.data.Dataset for training.

    Output:
        (spatial, labels_dict)

    labels_dict:
        {
            "binary": float32 scalar,
            "count_probs":  vector (17,),
            "slice":  vector (16,)
        }
    """

    spatial_sample, label_sample = _probe_sample(dataset_obj)

    spatial_spec = tf.TensorSpec(
        shape=tuple(spatial_sample.shape),
        dtype=tf.float32,
    )

    label_spec = {
        "binary": tf.TensorSpec(shape=(), dtype=tf.float32),
        "count_probs": tf.TensorSpec(shape=(17,), dtype=tf.float32),
        "slice": tf.TensorSpec(shape=(16,), dtype=tf.float32),
    }

    output_signature = (
        spatial_spec,
        label_spec,
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = (
        tf.data.Dataset
        .from_generator(
            lambda: _train_python_gen(dataset_obj),
            output_signature=output_signature,
        )
        .map(lambda s, l, a, b: (s, l), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds


def build_tf_val_dataset(dataset_obj, batch_size):
    """
    Build a finite validation dataset.

    Output matches training exactly for consistency.
    """

    if len(dataset_obj) == 0:
        raise RuntimeError("Validation dataset has zero length")

    first_spatial, first_label = _probe_sample(dataset_obj)

    spatial_spec = tf.TensorSpec(
        shape=tuple(first_spatial.shape),
        dtype=tf.float32,
    )

    label_spec = {
        "binary": tf.TensorSpec(shape=(), dtype=tf.float32),
        "count_probs": tf.TensorSpec(shape=(17,), dtype=tf.float32),
        "slice": tf.TensorSpec(shape=(16,), dtype=tf.float32),
    }

    output_signature = (
        spatial_spec,
        label_spec,
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: _val_python_gen(dataset_obj),
        output_signature=output_signature,
    )

    ds = (
        ds.map(lambda s, l, a, b: (s, l), num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return ds