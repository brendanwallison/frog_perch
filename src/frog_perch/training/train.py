# training/train.py
import os
import numpy as np
import tensorflow as tf

from frog_perch.datasets.frog_dataset import FrogPerchDataset
from frog_perch.models.downstream import build_downstream
import frog_perch.config as config

def python_gen(dataset_obj):
    """
    Generator that yields items from dataset_obj forever (random sampling inside dataset).
    Each item = (spatial_emb, label, audio_file, start)
    """
    while True:
        item = dataset_obj[np.random.randint(0, len(dataset_obj))]
        yield item  # (spatial_emb, label, audio_file, start)

def build_tf_dataset(dataset_obj, batch_size):
    # Probe a single element to create correct TensorSpecs
    sample = dataset_obj[0]
    spatial_shape = sample[0].shape  # (16,4,1536)
    label = sample[1]
    # label shape may be scalar or vector
    if np.isscalar(label):
        label_shape = ()
        label_dtype = np.float32
    else:
        label_shape = label.shape
        label_dtype = np.float32

    output_signature = (
        tf.TensorSpec(shape=spatial_shape, dtype=tf.float32),
        tf.TensorSpec(shape=label_shape, dtype=label_dtype),
        tf.TensorSpec(shape=(), dtype=tf.string),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_generator(lambda: python_gen(dataset_obj), output_signature=output_signature)
    ds = ds.map(lambda s,l,a,b: (s,l), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def train(label_mode=config.LABEL_MODE, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, perch_savedmodel_path=None):
    # instantiate Python dataset objects for train & val
    train_ds_obj = FrogPerchDataset(audio_dir=config.AUDIO_DIR, annotation_dir=config.ANNOTATION_DIR,
                                    train=True, pos_ratio=config.POS_RATIO, random_seed=config.RANDOM_SEED,
                                    label_mode=label_mode, perch_savedmodel_path=perch_savedmodel_path)
    val_ds_obj = FrogPerchDataset(audio_dir=config.AUDIO_DIR, annotation_dir=config.ANNOTATION_DIR,
                                  train=False, pos_ratio=config.POS_RATIO, random_seed=config.RANDOM_SEED,
                                  label_mode=label_mode, perch_savedmodel_path=perch_savedmodel_path)

    train_ds = build_tf_dataset(train_ds_obj, batch_size=batch_size)
    val_ds = build_tf_dataset(val_ds_obj, batch_size=batch_size)

    # model
    model = build_downstream(spatial_shape=(16,4,1536), label_mode=label_mode, pool_method='mean')

    if label_mode == 'binary':
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss=loss, metrics=[tf.keras.metrics.BinaryAccuracy()])
    else:
        loss = tf.keras.losses.KLDivergence()
        model.compile(optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE), loss=loss, metrics=[tf.keras.metrics.CategoricalAccuracy()])

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, 'downstream_best.h5')
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    ]

    # Steps: tune to size. default values chosen to be conservative.
    steps_per_epoch = 200
    validation_steps = 50

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks,
              steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)
    return model
