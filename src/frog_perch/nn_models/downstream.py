import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, regularizers


@keras.saving.register_keras_serializable(package="FrogPerch")
class DropFrequencyRows(layers.Layer):
    """
    Slices the frequency axis (index 1) to keep only rows 1 and 2.
    Replaces the inline lambda: lambda t: t[:, :, 1:3, :]
    """
    def call(self, inputs):
        return inputs[:, :, 1:3, :]


@keras.saving.register_keras_serializable(package="FrogPerch")
class BinaryFromCountProbs(layers.Layer):
    """
    Computes presence probability (1.0 - P(count=0)).
    Replaces the inline lambda: lambda cp: 1.0 - cp[:, 0]
    """
    def call(self, count_probs):
        return 1.0 - count_probs[:, 0]


@keras.saving.register_keras_serializable(package="FrogPerch")
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
                [
                    tf.zeros((B, 1), tf.float32),
                    dp[:, :-1],
                ],
                axis=1,
            )

            dp_new = dp * (1.0 - p_t) + shifted * p_t

            # Accumulate mass that would shift beyond max_bin
            dp_new = tf.concat(
                [
                    dp_new[:, :-1],
                    dp_new[:, -1:] + (dp[:, -1:] * p_t),
                ],
                axis=1,
            )

            return dp_new

        dp_final = tf.scan(step, probs_TB, initializer=dp0)
        return dp_final[-1]

    def get_config(self):
        config = super().get_config()
        config.update({"max_bin": self.max_bin})
        return config


@keras.saving.register_keras_serializable(package="FrogPerch")
class LocalExpansion(layers.Layer):
    """
    Learned temporal upsampling by local reshape.

    Splits each input timestep into `factor` sub-slices using only
    the channels allocated to that timestep — no cross-boundary
    interpolation.

    Input:  [B, T, factor * d]
    Output: [B, T * factor, d]

    The trunk must produce temporal_dim = factor * d channels so that
    each timestep carries exactly enough information for its sub-slices.
    """

    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = int(factor)

    def call(self, x):
        B  = tf.shape(x)[0]
        T  = tf.shape(x)[1]
        C  = tf.shape(x)[2]
        d  = C // self.factor

        # [B, T, factor*d] -> [B, T, factor, d]
        x = tf.reshape(x, (B, T, self.factor, d))

        # [B, T, factor, d] -> [B, T*factor, d]
        x = tf.reshape(x, (B, T * self.factor, d))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({"factor": self.factor})
        return config


def build_downstream(
    spatial_shape=(16, 4, 1536),
    slice_hidden_dims=(1024, 512),
    temporal_dim=384,           # must be divisible by (n_slices // T_in)
    num_temporal_layers=2,
    kernel_size=3,
    activation="gelu",
    dropout=0.1,
    l2_reg=1e-4,
    use_gating=False,
    max_bin=8,
    n_slices=48,                # must be an integer multiple of T_in
    drop_freq_rows=True,
):
    """
    Outputs:
        - slice         [B, n_slices]
        - slice_probs   [B, n_slices]
        - count_probs   [B, max_bin+1]
        - binary        [B]
        - count_logits  [B, n_slices]  passthrough for visualizer
    """

    T_in, F, C = spatial_shape

    # 1. Handle Frequency Slicing
    if drop_freq_rows:
        assert F == 4, "drop_freq_rows requires frequency dimension to be 4"
        # Slice frequency axis (index 1), keep rows 1 and 2 (the middle ones)
        # Input shape becomes (16, 2, 1536)
        spatial_shape_eff = (T_in, 2, C)
        F_eff = 2
    else:
        spatial_shape_eff = spatial_shape
        F_eff = F

    FLAT_DIM = F_eff * C

    # ------------------------------------------------------------------
    # Expansion constraints
    # ------------------------------------------------------------------
    assert n_slices % T_in == 0, (
        f"n_slices ({n_slices}) must be an integer multiple of T_in ({T_in})"
    )
    factor = n_slices // T_in
    assert temporal_dim % factor == 0, (
        f"temporal_dim ({temporal_dim}) must be divisible by factor ({factor}); "
        f"got temporal_dim={temporal_dim}, factor={factor}"
    )
    sub_dim = temporal_dim // factor   # channel dim after expansion

    inp = layers.Input(
        shape=spatial_shape,
        dtype=tf.float32,
        name="spatial_emb",
    )

    # Apply the slice toggle
    if drop_freq_rows:
        # Keep middle 2 rows of the frequency dimension (index 1)
        x = DropFrequencyRows(name="drop_freq_rows")(inp)
    else:
        x = inp

    # ------------------------------------------------------------------
    # 1. Per-slice encoding
    # ------------------------------------------------------------------

    x = layers.Reshape((T_in, FLAT_DIM))(x)

    for dim in slice_hidden_dims:
        x = layers.TimeDistributed(
            layers.Dense(
                dim,
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )(x)

        x = layers.LayerNormalization()(x)
        x = layers.SpatialDropout1D(dropout / 2.0)(x)

    x = layers.TimeDistributed(
        layers.Dense(
            temporal_dim,
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
        )
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.SpatialDropout1D(dropout)(x)

    # ------------------------------------------------------------------
    # 2. LOCAL TEMPORAL MIXING AT NATIVE RESOLUTION
    #
    # Keep receptive field intentionally small.
    # This stage extracts local temporal structure from the pretrained
    # embeddings before any expansion occurs.
    # ------------------------------------------------------------------

    for _ in range(num_temporal_layers):
        residual = x

        norm = layers.LayerNormalization()(x)

        h = layers.Conv1D(
            filters=temporal_dim,
            kernel_size=kernel_size,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
        )(norm)

        h = layers.Dropout(dropout)(h)

        if use_gating:
            gate = layers.Conv1D(
                filters=temporal_dim,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",
                kernel_regularizer=regularizers.l2(l2_reg),
            )(norm)

            h = layers.Multiply()([gate, h])

        x = layers.Add()([residual, h])

    # ------------------------------------------------------------------
    # 3. LEARNED LOCAL EXPANSION
    #
    # Each input timestep is split into `factor` sub-slices purely from
    # its own channels — no cross-boundary interpolation.
    # [B, T_in, temporal_dim] -> [B, n_slices, sub_dim]
    # ------------------------------------------------------------------

    x = LocalExpansion(factor=factor, name="local_expansion")(x)

    # Project back up to temporal_dim so refinement convolutions have
    # full capacity.
    x = layers.Dense(
        temporal_dim,
        activation=activation,
        kernel_regularizer=regularizers.l2(l2_reg),
        name="expansion_proj",
    )(x)

    x = layers.LayerNormalization()(x)

    # ------------------------------------------------------------------
    # 4. SHALLOW LOCAL REFINEMENT AT TARGET RESOLUTION
    #
    # Allows limited information sharing across sub-slice boundaries
    # via small receptive field convolutions.
    # ------------------------------------------------------------------

    for _ in range(3):
        residual = x

        h = layers.LayerNormalization()(x)

        h = layers.Conv1D(
            filters=temporal_dim,
            kernel_size=3,
            padding="same",
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg),
        )(h)

        h = layers.Dropout(dropout / 2.0)(h)

        x = layers.Add()([residual, h])

    x = layers.LayerNormalization()(x)

    # ------------------------------------------------------------------
    # A. COUNTING BRANCH
    # ------------------------------------------------------------------

    count_logits = layers.TimeDistributed(
        layers.Dense(
            1,
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        name="count_dense",
    )(x)

    count_logits = layers.Flatten()(count_logits)

    count_probs_t = layers.Activation(
        "sigmoid",
        name="count_slice_probs",
    )(count_logits)

    count_probs = SoftCountFromSlices(
        max_bin=max_bin,
        name="count_probs",
    )(count_probs_t)

    binary_prob = BinaryFromCountProbs(
        name="binary",
    )(count_probs)

    # ------------------------------------------------------------------
    # B. SLICE CLASSIFICATION BRANCH
    # ------------------------------------------------------------------

    slice_logits = layers.TimeDistributed(
        layers.Dense(
            1,
            kernel_regularizer=regularizers.l2(l2_reg),
        ),
        name="slice_dense",
    )(x)

    slice_output = layers.Flatten(name="slice")(slice_logits)

    return Model(
        inputs=inp,
        outputs={
            "slice":        slice_output,
            "count_probs":  count_probs,
            "binary":       binary_prob,
        },
        name="frog_perch_local_interpolation",
    )