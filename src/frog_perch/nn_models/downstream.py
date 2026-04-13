import tensorflow as tf
from tensorflow.keras import layers, Model, regularizers


class SoftCountFromSlices(layers.Layer):
    """
    Poisson-binomial distribution over slice Bernoulli probabilities.
    Output: [B, max_bin + 1]
    """

    def __init__(self, max_bin=16, **kwargs):
        super().__init__(**kwargs)
        self.max_bin = int(max_bin)

    def call(self, slice_probs):
        slice_probs = tf.convert_to_tensor(slice_probs, tf.float32)

        B = tf.shape(slice_probs)[0]

        dp0 = tf.concat(
            [
                tf.ones((B, 1), tf.float32),
                tf.zeros((B, self.max_bin), tf.float32),
            ],
            axis=1,
        )

        probs_TB = tf.transpose(slice_probs, [1, 0])  # [T, B]

        def step(dp, p_t):
            p_t = tf.expand_dims(p_t, 1)

            shifted = tf.concat(
                [tf.zeros((B, 1), tf.float32), dp[:, :-1]],
                axis=1,
            )

            dp_new = dp * (1.0 - p_t) + shifted * p_t

            # overflow accumulation (stable)
            dp_new = tf.concat(
                [
                    dp_new[:, :-1],
                    dp_new[:, -1:] + dp[:, -1:],
                ],
                axis=1,
            )

            return dp_new

        dp_final = tf.scan(step, probs_TB, initializer=dp0)
        return dp_final[-1]


def build_downstream(
    spatial_shape=(16, 4, 1536),
    slice_hidden_dims=(512, 128),  # Adjust this tuple to experiment with dimensionality reduction
    temporal_dim=128,
    num_temporal_layers=2,
    kernel_size=3,
    dropout=0.1,
    l2_reg=1e-4,                   # NEW: L2 weight decay parameter
    use_gating=False,
    max_bin=16,
):
    """
    Outputs:
        - slice_logits  [B, T]
        - slice_probs   [B, T]
        - count_probs   [B, max_bin+1]
        - binary_prob   [B, 1]
    """

    T, F, C = spatial_shape
    FLAT_DIM = F * C

    inp = layers.Input(shape=spatial_shape, dtype=tf.float32, name="spatial_emb")

    # ------------------------------------------------------------
    # Per-slice encoding (with Intermediate Normalization & L2)
    # ------------------------------------------------------------
    x = layers.Reshape((T, FLAT_DIM))(inp)

    for dim in slice_hidden_dims:
        x = layers.TimeDistributed(
            layers.Dense(
                dim, 
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.SpatialDropout1D(dropout / 2.0)(x) # Light dropout between dense layers

    x = layers.TimeDistributed(
        layers.Dense(
            temporal_dim, 
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        )
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)

    # ------------------------------------------------------------
    # Temporal mixing (with Pre-Norm & L2)
    # ------------------------------------------------------------
    for _ in range(num_temporal_layers):
        
        # Pre-Norm: Normalize the input to the residual branch
        res_inp = layers.LayerNormalization()(x)
        
        h = layers.Conv1D(
            filters=temporal_dim,
            kernel_size=kernel_size,
            padding="same",
            activation="relu",
            kernel_regularizer=regularizers.l2(l2_reg)
        )(res_inp)

        h = layers.Dropout(dropout)(h)

        if use_gating:
            gate = layers.Conv1D(
                filters=temporal_dim,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                kernel_regularizer=regularizers.l2(l2_reg)
            )(res_inp)
            x = layers.Add()([x, layers.Multiply()([gate, h])])
        else:
            x = layers.Add()([x, h])

    # ------------------------------------------------------------
    # Slice head
    # ------------------------------------------------------------
    x = layers.LayerNormalization()(x)

    slice_logits = layers.TimeDistributed(
        layers.Dense(
            1, 
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        name="slice_logits"
    )(x)

    slice_logits = layers.Reshape((T,), name="slice_logits_flat")(slice_logits)
    slice_probs = layers.Activation("sigmoid", name="slice_probs")(slice_logits)
    count_probs = SoftCountFromSlices(max_bin=max_bin, name="count_probs")(slice_probs)
    binary_prob = layers.Lambda(
        lambda p: 1.0 - p[:, 0:1],  # 1.0 - Probability of 0 count
    )(count_probs)

    return Model(
        inputs=inp,
        outputs={
            "slice_logits": slice_logits,
            "slice_probs": slice_probs,
            "count_probs": count_probs,
            "binary": binary_prob,
        },
        name="frog_perch_downstream",
    )