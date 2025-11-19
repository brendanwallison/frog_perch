# models/downstream.py
import tensorflow as tf
from keras import layers, Model
import keras.ops as ops

class AttentionPool2D(layers.Layer):
    """
    Lightweight spatial attention:
      score = tanh(Wx)
      weights = softmax(score)
      pooled = sum(weights * x)
    """

    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation='tanh')
        self.dense2 = layers.Dense(1)

    def call(self, x):
        # x: [B,H,W,C]
        B, H, W, C = x.shape  # may contain None — fine

        # score = tanh(Wx)
        score = self.dense1(x)          # [B,H,W,U]
        score = self.dense2(score)      # [B,H,W,1]

        # ---- FIX: reshape to apply softmax over one axis ----
        # flatten spatial dims
        score_flat = ops.reshape(score, (ops.shape(x)[0], -1))   # [B, H*W]

        # softmax over flattened spatial dimension
        weights_flat = ops.softmax(score_flat, axis=-1)          # [B, H*W]

        # reshape back to spatial shape
        weights = ops.reshape(weights_flat, (ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], 1))
        # ------------------------------------------------------

        # pooled = sum(weights * x)
        pooled = ops.sum(x * weights, axis=(1, 2))  # [B,C]

        return pooled

    def compute_output_shape(self, input_shape):
        # return [B, C]
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


def build_downstream(
    spatial_shape=(16, 4, 1536),
    label_mode='count',           # 'binary' or 'count'
    pool_method='mean'            # one of:
                                  # 'mean', 'conv', 'avgmax', 'mlp_flat',
                                  # 'attn', 'conv2', 'bottleneck1x1',
                                  # 'temporal', 'freq'
):
    inp = layers.Input(shape=spatial_shape, dtype=tf.float32, name='spatial_emb')
    x = inp

    ## ---------------------------------------------------------
    ## 1. MEAN POOLING (baseline)
    ## ---------------------------------------------------------
    if pool_method == 'mean':
        x = layers.GlobalAveragePooling2D()(x)  # → [B,1536]

    ## ---------------------------------------------------------
    ## 2. CONV → POOL (your original conv option)
    ## ---------------------------------------------------------
    elif pool_method == 'conv':
        x = layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)

    ## ---------------------------------------------------------
    ## 3. AVGMAX POOLING (concatenate avg + max)
    ## ---------------------------------------------------------
    elif pool_method == 'avgmax':
        avg = layers.GlobalAveragePooling2D()(x)          # [B,C]
        mx  = layers.GlobalMaxPooling2D()(x)              # [B,C]
        x = layers.Concatenate()([avg, mx])               # [B,2C]

    ## ---------------------------------------------------------
    ## 4. MLP on full embedding (flatten → bottleneck → MLP)
    ## ---------------------------------------------------------
    elif pool_method == 'mlp_flat':
        x = layers.Flatten()(x)                           # [B, ~100k]
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)

    ## ---------------------------------------------------------
    ## 5. ATTENTION POOLING (learned weighted pooling)
    ## ---------------------------------------------------------
    elif pool_method == 'attn':
        x = AttentionPool2D(units=256)(x)              # [B,C]

    ## ---------------------------------------------------------
    ## 6. CONV → CONV → POOL (deeper adapter)
    ## ---------------------------------------------------------
    elif pool_method == 'conv2':
        x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)

    ## ---------------------------------------------------------
    ## 7. 1×1 CONV bottleneck before pooling
    ## ---------------------------------------------------------
    elif pool_method == 'bottleneck1x1':
        x = layers.Conv2D(256, (1,1), activation='relu')(x)   # reduce channels
        x = layers.GlobalAveragePooling2D()(x)

    ## ---------------------------------------------------------
    ## 8. TEMPORAL POOL only (pool over H, keep W structure)
    ## ---------------------------------------------------------
    elif pool_method == 'temporal':
        x = tf.reduce_mean(x, axis=1)                     # [B,W,C]
        x = layers.Flatten()(x)                           # [B, W*C]
        x = layers.Dense(512, activation='relu')(x)

    ## ---------------------------------------------------------
    ## 9. FREQ POOL only (pool over W)
    ## ---------------------------------------------------------
    elif pool_method == 'freq':
        x = tf.reduce_mean(x, axis=2)                     # [B,H,C]
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)

    ## ---------------------------------------------------------
    ## Unknown option
    ## ---------------------------------------------------------
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")

    ## ---------------------------------------------------------
    ## Shared MLP head (unless mlp_flat already handled it)
    ## ---------------------------------------------------------
    if pool_method not in ['mlp_flat', 'temporal', 'freq']:
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

    ## ---------------------------------------------------------
    ## OUTPUT: binary or count distribution
    ## ---------------------------------------------------------
    if label_mode == 'binary':
        logits = layers.Dense(1, name='logit')(x)
        model = Model(inputs=inp, outputs=logits)
        return model

    else:
        logits = layers.Dense(5, name='count_logits')(x)
        probs = layers.Activation('softmax')(logits)
        model = Model(inputs=inp, outputs=probs)
        return model

