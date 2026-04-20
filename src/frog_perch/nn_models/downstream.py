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


# def build_downstream(
#     spatial_shape=(16, 4, 1536),
#     slice_hidden_dims=(1024, 512),  
#     temporal_dim=128,
#     num_temporal_layers=2,
#     num_heads=8,                 # 128 / 8 = 16 dimensions per head
#     activation="gelu",
#     dropout=0.1,
#     l2_reg=1e-4,                   
#     max_bin=16,
#     **kwargs                     # Captures unused kernel_size/use_gating
# ):
#     """
#     Transformer-based mixing for global window context.
#     Outputs:
#         - slice: [B, T] (logits)
#         - slice_probs: [B, T] (sigmoid)
#         - count_probs: [B, max_bin+1] (PMF)
#         - binary: [B, 1] (presence prob)
#     """

#     T, F, C = spatial_shape
#     FLAT_DIM = F * C

#     inp = layers.Input(shape=spatial_shape, dtype=tf.float32, name="spatial_emb")

#     # 1. Per-slice encoding (Identical to your Conv version)
#     x = layers.Reshape((T, FLAT_DIM))(inp)

#     for dim in slice_hidden_dims:
#         x = layers.TimeDistributed(
#             layers.Dense(
#                 dim, 
#                 activation=activation,
#                 kernel_regularizer=regularizers.l2(l2_reg)
#             )
#         )(x)
#         x = layers.LayerNormalization()(x)
#         x = layers.SpatialDropout1D(dropout / 2.0)(x)

#     x = layers.TimeDistributed(
#         layers.Dense(
#             temporal_dim, 
#             activation=activation,
#             kernel_regularizer=regularizers.l2(l2_reg)
#         )
#     )(x)
#     x = layers.LayerNormalization()(x)

#     # ------------------------------------------------------------
#     # 2. Positional Encoding
#     # ------------------------------------------------------------
#     # Since Transformers are permutation-invariant, we must inject 
#     # the temporal order of the 16 slices.
#     positions = tf.range(start=0, limit=T, delta=1)
#     pos_emb = layers.Embedding(input_dim=T, output_dim=temporal_dim)(positions)
#     x = x + pos_emb 

#     # ------------------------------------------------------------
#     # 3. Transformer Mixing (Residual Global Attention)
#     # ------------------------------------------------------------
#     for _ in range(num_temporal_layers):
        
#         # A. Multi-Head Self-Attention (Global context)
#         res_inp = layers.LayerNormalization()(x)
#         attn_out = layers.MultiHeadAttention(
#             num_heads=num_heads, 
#             key_dim=temporal_dim // num_heads,
#             kernel_regularizer=regularizers.l2(l2_reg)
#         )(res_inp, res_inp)  # Query=res_inp, Value=res_inp
        
#         attn_out = layers.Dropout(dropout)(attn_out)
#         x = layers.Add()([x, attn_out]) # First Residual

#         # B. Feed-Forward (No expansion, just feature refinement)
#         res_inp = layers.LayerNormalization()(x)
#         ffn_out = layers.Dense(
#             temporal_dim, 
#             activation=activation,
#             kernel_regularizer=regularizers.l2(l2_reg)
#         )(res_inp)
        
#         ffn_out = layers.Dropout(dropout)(ffn_out)
#         x = layers.Add()([x, ffn_out]) # Second Residual

#     # ------------------------------------------------------------
#     # 4. Output Heads (Linked via TimeDistributed)
#     # ------------------------------------------------------------
#     x = layers.LayerNormalization()(x)

#     # Link to Slice Predictions: 
#     # Each of the 16 refined temporal vectors (shape 128) is projected to a single logit.
#     slice_logits = layers.TimeDistributed(
#         layers.Dense(
#             1, 
#             kernel_regularizer=regularizers.l2(l2_reg)
#         ),
#         name="slice_dense" 
#     )(x)
    
#     slice_output = layers.Reshape((T,), name="slice")(slice_logits)
#     slice_probs = layers.Activation("sigmoid", name="slice_probs")(slice_output)

#     # Count & Binary heads remain identical
#     count_probs = SoftCountFromSlices(max_bin=max_bin, name="count_probs")(slice_probs)
#     prob_zero = count_probs[:, 0:1]
#     binary_prob = 1.0 - prob_zero 

#     return Model(
#         inputs=inp,
#         outputs={
#             "slice": slice_output,
#             "slice_probs": slice_probs,
#             "count_probs": count_probs,
#             "binary": binary_prob,
#         },
#         name="frog_perch_transformer",
#     )

def build_downstream(
    spatial_shape=(16, 4, 1536),
    slice_hidden_dims=(1024, 512),  
    temporal_dim=128,
    num_temporal_layers=2,
    kernel_size=3,
    activation="gelu",
    dropout=0.1,
    l2_reg=1e-4,                   
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
                activation=activation,
                kernel_regularizer=regularizers.l2(l2_reg)
            )
        )(x)
        x = layers.LayerNormalization()(x)
        x = layers.SpatialDropout1D(dropout / 2.0)(x) # Light dropout between dense layers

    x = layers.TimeDistributed(
        layers.Dense(
            temporal_dim, 
            activation=activation,      # UPDATED
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
            activation=activation,
            kernel_regularizer=regularizers.l2(l2_reg)
        )(res_inp)

        h = layers.Dropout(dropout)(h)

        if use_gating:
            gate = layers.Conv1D(
                filters=temporal_dim,
                kernel_size=kernel_size,
                padding="same",
                activation="sigmoid",   # sigmoid for gating logic [0, 1]
                kernel_regularizer=regularizers.l2(l2_reg)
            )(res_inp)
            x = layers.Add()([x, layers.Multiply()([gate, h])])
        else:
            x = layers.Add()([x, h])

    # ------------------------------------------------------------
    # Output Heads
    # ------------------------------------------------------------
    x = layers.LayerNormalization()(x)

    # 1. Slice Head (Named "slice" to match dataset labels)
    # We produce raw logits here for numerical stability in the loss function,
    # but the head is named semantically.
    slice_logits = layers.TimeDistributed(
        layers.Dense(
            1, 
            kernel_regularizer=regularizers.l2(l2_reg)
        ),
        name="slice_dense" 
    )(x)
    
    # Final slice output: [B, T]
    slice_output = layers.Reshape((T,), name="slice")(slice_logits)

    # 2. Probability Conversion (Internal)
    # Needed for the Poisson-Binomial layer, even if not used as a primary loss target.
    slice_probs = layers.Activation("sigmoid", name="slice_probs")(slice_output)

    # 3. Count Head: Poisson-Binomial Distribution [B, max_bin + 1]
    count_probs = SoftCountFromSlices(max_bin=max_bin, name="count_probs")(slice_probs)
    
    # 4. Binary Head: Probability of at least one call [B]
    prob_zero = count_probs[:, 0] # Removed the 0:1 slice to drop the extra dimension
    binary_prob = 1.0 - prob_zero

    return Model(
        inputs=inp,
        outputs={
            "slice": slice_output,        # Binary targets (0/1) go here
            "slice_probs": slice_probs,   # For visualization/inference
            "count_probs": count_probs,   # PMF targets go here
            "binary": binary_prob,        # Presence/Absence targets go here
        },
        name="frog_perch_downstream",
    )