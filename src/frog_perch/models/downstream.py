# models/downstream.py
import tensorflow as tf
from keras import layers, Model
import keras.ops as ops
from typing import List, Optional

# ---------------------------------------------------------------------------
# Utility: pure-TF soft count distribution (modular)
# ---------------------------------------------------------------------------

class SoftCountFromSlices(layers.Layer):
    """
    Convert per-slice Bernoulli probabilities (shape [B, T]) into a soft count
    distribution truncated at max_bin (0..max_bin, where max_bin bin accumulates
    counts >= max_bin). Implements the exact logic of the reference Python
    soft_count_distribution function in pure TensorFlow.
    """

    def __init__(self, max_bin: int = 4, **kwargs):
        super().__init__(**kwargs)
        self.max_bin = int(max_bin)
        assert self.max_bin >= 0

    def build(self, input_shape):
        # nothing to build
        super().build(input_shape)

    def call(self, probs: tf.Tensor) -> tf.Tensor:
        """
        probs: tf.Tensor of shape [B, T] (float32 in [0,1])
        returns: tf.Tensor of shape [B, max_bin+1] (float32), each row sums to 1
        """

        max_bin = self.max_bin
        dtype = tf.float32

        # Ensure float32
        probs = tf.cast(probs, dtype)

        # Per-example function: fold over temporal probs
        def _single_soft_count(single_probs):
            # single_probs: [T]
            # Initial distribution: dist[0]=1, rest 0
            init = tf.concat([[tf.constant(1.0, dtype=dtype)],
                              tf.zeros([max_bin], dtype=dtype)], axis=0)  # shape [max_bin+1]

            # Update function for one p
            def _update(dist, p):
                # Clip p
                p = tf.clip_by_value(p, 0.0, 1.0)

                # If p == 0.0 -> return dist unchanged
                def _case_zero():
                    return dist

                # If p == 1.0 -> shift dist right by 1, and add last bin accumulation
                def _case_one():
                    # shifted: new[1:] = dist[:-1], new[0]=0
                    shifted = tf.concat([tf.zeros([1], dtype=dtype), dist[:-1]], axis=0)
                    # add dist[-1] into last bin
                    shifted = tf.tensor_scatter_nd_add(shifted, [[max_bin]], [dist[-1]])
                    return shifted

                # General case 0 < p < 1
                def _case_general():
                    # new = dist * (1 - p)
                    new = dist * (1.0 - p)
                    # add dist[:-1] * p to new[1:]
                    add_shift = tf.concat([tf.zeros([1], dtype=dtype), dist[:-1] * p], axis=0)
                    new = new + add_shift
                    # add dist[-1] * p to new[-1]
                    new = tf.tensor_scatter_nd_add(new, [[max_bin]], [dist[-1] * p])
                    return new

                # Branch on p
                is_zero = tf.equal(p, 0.0)
                is_one = tf.equal(p, 1.0)

                # Use nested conds to match original logic order
                return tf.cond(is_zero,
                               lambda: _case_zero(),
                               lambda: tf.cond(is_one, _case_one, _case_general))

            # Fold (left) across the sequence of probs
            # tf.foldl applies fn(acc, elem) across elems
            final = tf.foldl(lambda d, pp: _update(d, pp), single_probs, initializer=init)

            # Normalize (if sum > 0) else set dist[0]=1
            s = tf.reduce_sum(final)
            final = tf.cond(s > 0.0,
                            lambda: final / s,
                            lambda: tf.concat([[tf.constant(1.0, dtype=dtype)],
                                               tf.zeros([max_bin], dtype=dtype)], axis=0))
            return final  # shape [max_bin+1]

        # Handle empty temporal axis (T==0): map to [1,0,...]
        # tf.map_fn will handle batch dimension; if T==0, single_probs will be shape (0,)
        # and foldl will return initializer (works).
        out = tf.map_fn(_single_soft_count, probs, dtype=dtype, parallel_iterations=8)
        # out shape: [B, max_bin+1]
        return out

    def compute_output_shape(self, input_shape):
        # input_shape: (B, T)
        return (input_shape[0], self.max_bin + 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"max_bin": self.max_bin})
        return cfg

# ---------------------------------------------------------------------------
# Existing pooling layers (unchanged) - include AttentionPool2D and Multihead...
# ---------------------------------------------------------------------------

class MultiheadAttentionPool2D(layers.Layer):
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=0.0)

    def build(self, input_shape):
        _, H, W, C = input_shape
        self.cls_token = self.add_weight(shape=(1, 1, C), initializer="zeros", trainable=True, name="cls_token")

    def call(self, x):
        B = ops.shape(x)[0]
        H = ops.shape(x)[1]
        W = ops.shape(x)[2]
        C = ops.shape(x)[3]
        x_flat = ops.reshape(x, (B, H * W, C))
        cls = ops.broadcast_to(self.cls_token, (B, 1, C))
        x_in = ops.concatenate([cls, x_flat], axis=1)
        attended = self.mha(query=x_in, value=x_in, key=x_in, training=self.trainable)
        pooled = attended[:, 0, :]
        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return cfg


class AttentionPool2D(layers.Layer):
    def __init__(self, units=256, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense1 = layers.Dense(units, activation='tanh')
        self.dense2 = layers.Dense(1)

    def call(self, x):
        B, H, W, C = x.shape
        score = self.dense1(x)
        score = self.dense2(score)
        score_flat = ops.reshape(score, (ops.shape(x)[0], -1))
        weights_flat = ops.softmax(score_flat, axis=-1)
        weights = ops.reshape(weights_flat, (ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2], 1))
        pooled = ops.sum(x * weights, axis=(1, 2))
        return pooled

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ---------------------------------------------------------------------------
# build_downstream: integrates the modular SoftCountFromSlices layer
# ---------------------------------------------------------------------------

def build_downstream(
    spatial_shape=(16, 4, 1536),
    label_mode='count',           # 'binary', 'count', or 'slice'
    pool_method='mean'            # existing pool options
):
    inp = layers.Input(shape=spatial_shape, dtype=tf.float32, name='spatial_emb')
    x = inp

    # Slice-based architecture: per-temporal-slice binary heads
    if pool_method == "slice":

        TIME_SLICES = 16  # must match spatial_shape[0]
        FREQ = spatial_shape[1]      # 4
        CHANNELS = spatial_shape[2]  # 1536
        FLAT_DIM = FREQ * CHANNELS   # 6144

        # x: [B, T, F, C] = [B, 16, 4, 1536]
        # Flatten freq × channels per slice
        x_flat = layers.Reshape((TIME_SLICES, FLAT_DIM))(x)  # [B, 16, 6144]

        # --- Per-slice MLP projection ---
        slice_embed = layers.TimeDistributed(
            layers.Dense(1024, activation="relu")
        )(x_flat)

        slice_embed = layers.TimeDistributed(
            layers.Dense(512, activation="relu")
        )(slice_embed)  # [B, 16, 512]

        # Optional dropout for regularization
        slice_embed = layers.Dropout(0.2)(slice_embed)

        # --- Temporal neighborhood via Conv1D with residual ---
        # Conv1D operates over time: [B, T, 512]
        temporal_context = layers.Conv1D(
            filters=512,
            kernel_size=3,
            padding="same",
            activation="relu"
        )(slice_embed)

        # Residual connection: model can ignore neighbors if unhelpful
        slice_embed = layers.Add()([slice_embed, temporal_context])  # [B, 16, 512]

        # Optional second temporal layer (deeper receptive field)
        temporal_context2 = layers.Conv1D(
            filters=512,
            kernel_size=3,
            padding="same",
            activation="relu"
        )(slice_embed)

        slice_embed = layers.Add()([slice_embed, temporal_context2])  # [B, 16, 512]

        # --- Final per-slice logits ---
        slice_logits = layers.TimeDistributed(
            layers.Dense(1),
            name="slice_logits"
        )(slice_embed)  # [B, 16, 1]

        # Flatten to [B, 16]
        slice_logits = layers.Reshape((TIME_SLICES,), name="slice_logits_flat")(slice_logits)

        model = Model(inputs=inp, outputs=slice_logits)
        return model

    # Otherwise, existing pooling options (unchanged)
    if pool_method == 'mean':
        x = layers.GlobalAveragePooling2D()(x)
    elif pool_method == 'conv':
        x = layers.Conv2D(512, (3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif pool_method == 'avgmax':
        avg = layers.GlobalAveragePooling2D()(x)
        mx  = layers.GlobalMaxPooling2D()(x)
        x = layers.Concatenate()([avg, mx])
    elif pool_method == 'mlp_flat':
        x = layers.Flatten()(x)
        x = layers.Dense(2048, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
    elif pool_method == 'attn':
        x = AttentionPool2D(units=256)(x)
    elif pool_method == 'conv2':
        x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3,3), padding='same', activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif pool_method == 'bottleneck1x1':
        x = layers.Conv2D(256, (1,1), activation='relu')(x)
        x = layers.GlobalAveragePooling2D()(x)
    elif pool_method == 'temporal':
        x = tf.reduce_mean(x, axis=1)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
    elif pool_method == 'freq':
        x = tf.reduce_mean(x, axis=2)
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
    elif pool_method == 'mha':
        x = MultiheadAttentionPool2D(num_heads=4, key_dim=128)(x)
    else:
        raise ValueError(f"Unknown pool_method: {pool_method}")

    if pool_method not in ['mlp_flat', 'temporal', 'freq']:
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)

    if label_mode == 'binary':
        logits = layers.Dense(1, name='logit')(x)
        model = Model(inputs=inp, outputs=logits)
        return model
    elif label_mode == 'count':
        logits = layers.Dense(5, name='count_logits')(x)
        probs = layers.Activation('softmax')(logits)
        model = Model(inputs=inp, outputs=probs)
        return model 
    else:
        logits = layers.Dense(16, name='count_logits')(x)
        model = Model(inputs=inp, outputs=logits)
        return model
