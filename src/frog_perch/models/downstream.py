# models/downstream.py
import tensorflow as tf
from tensorflow.keras import layers, Model

def build_downstream(spatial_shape=(16,4,1536), label_mode='count', pool_method='mean'):
    inp = layers.Input(shape=spatial_shape, dtype=tf.float32, name='spatial_emb')
    x = inp
    if pool_method == 'mean':
        x = layers.GlobalAveragePooling2D()(x)  # -> [B,1536]
    elif pool_method == 'conv':
        x = layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    if label_mode == 'binary':
        logits = layers.Dense(1, name='logit')(x)
        model = Model(inputs=inp, outputs=logits)
        return model
    else:
        logits = layers.Dense(5, name='count_logits')(x)
        probs = layers.Activation('softmax')(logits)
        model = Model(inputs=inp, outputs=probs)
        return model
