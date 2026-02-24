import tensorflow as tf
import numpy as np

from frog_perch.metrics.count_metrics import (
    ExpectedCountMAE,
    EMD,
    ExpectedRecall,
    ExpectedPrecision,
    ExpectedBinaryAccuracy,
)

# ------------------------------------------------------------
# Pure-TensorFlow soft count distribution
# ------------------------------------------------------------

def tf_soft_count_distribution(weights, max_bin=4):
    """
    Pure TF version of soft_count_distribution.
    weights: [T] probabilities
    returns: [max_bin+1] soft count distribution
    """

    weights = tf.convert_to_tensor(weights, dtype=tf.float32)  # [T]
    T = tf.shape(weights)[0]

    # dp[k] = probability of exactly k events
    dp = tf.concat([tf.constant([1.0]), tf.zeros([max_bin], tf.float32)], axis=0)

    def body(i, dp):
        p = weights[i]  # scalar
        dp_shifted = tf.concat([tf.constant([0.0]), dp[:-1]], axis=0)
        dp_new = dp * (1.0 - p) + dp_shifted * p
        return i + 1, dp_new

    def cond(i, dp):
        return i < T

    _, dp_final = tf.while_loop(
        cond,
        body,
        loop_vars=[0, dp],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([None])]
    )

    return dp_final  # [max_bin+1]

# ------------------------------------------------------------
# Pure-TensorFlow soft count distribution via scan, T=16 fixed
# ------------------------------------------------------------

def tf_soft_count_distribution_batch(weights, max_bin=4, expected_T=16):
    """
    weights: [B, T] probabilities, with static T=expected_T
    returns: [B, max_bin+1] soft count distributions
    """
    weights = tf.convert_to_tensor(weights, dtype=tf.float32)  # [B, T]

    # Use *static* T from the shape, not dynamic
    static_shape = weights.shape          # TensorShape([None, 16]) if things are right
    T = static_shape[1]
    if T is None:
        raise ValueError(
            f"tf_soft_count_distribution_batch requires static T, "
            f"but got shape {static_shape}"
        )
    if T != expected_T:
        raise ValueError(
            f"Expected T={expected_T} slices, but got T={T} from shape {static_shape}"
        )

    B = tf.shape(weights)[0]

    # dp: [B, max_bin+1], initialize with P(K=0)=1
    dp0 = tf.concat(
        [tf.ones((B, 1), dtype=tf.float32), tf.zeros((B, max_bin), dtype=tf.float32)],
        axis=1,
    )

    def step(dp, p_t):
        # dp: [B, K+1], p_t: [B]
        p_t = tf.expand_dims(p_t, axis=1)  # [B, 1]
        dp_shifted = tf.concat(
            [tf.zeros((B, 1), dtype=tf.float32), dp[:, :-1]], axis=1
        )
        dp_new = dp * (1.0 - p_t) + dp_shifted * p_t
        return dp_new

    # Transpose to scan over time: [T, B]
    weights_TB = tf.transpose(weights, perm=[1, 0])  # [T, B]

    # T is static (expected_T), so scan has fixed length
    dp_final = tf.scan(
        fn=step,
        elems=weights_TB,   # each elem: [B]
        initializer=dp0,
    )[-1]  # take final state: [B, K+1]

    return dp_final  # [B, max_bin+1]


def slices_to_soft_counts(slice_logits, max_bin=4, expected_T=16):
    slice_probs = tf.nn.sigmoid(slice_logits)  # [B, T]
    return tf_soft_count_distribution_batch(slice_probs, max_bin=max_bin, expected_T=expected_T)


def targets_to_soft_counts(slice_targets, max_bin=4, expected_T=16):
    return tf_soft_count_distribution_batch(slice_targets, max_bin=max_bin, expected_T=expected_T)



# ------------------------------------------------------------
# Generic wrapper for clip-level metrics
# ------------------------------------------------------------

class SliceToCountWrapper(tf.keras.metrics.Metric):
    """
    Wraps a clip-level metric so it can be applied to slice-level logits.
    """
    def __init__(self, base_metric, max_bin=4, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_metric = base_metric
        self.max_bin = max_bin

    def update_state(self, y_true, y_pred, sample_weight=None):
        # (Assuming functions targets_to_soft_counts/slices_to_soft_counts are defined in this file)
        true_counts = targets_to_soft_counts(y_true, max_bin=self.max_bin)
        pred_counts = slices_to_soft_counts(y_pred, max_bin=self.max_bin)
        self.base_metric.update_state(true_counts, pred_counts, sample_weight)

    def result(self):
        return self.base_metric.result()

    def reset_state(self):
        self.base_metric.reset_state()

    def get_config(self):
        """Standard Keras serialization: save the arguments needed to recreate this class."""
        config = super().get_config()
        config.update({
            "base_metric": tf.keras.metrics.serialize(self.base_metric),
            "max_bin": self.max_bin,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """
        Custom deserialization to handle legacy checkpoints where 'base_metric' is missing.
        """
        # Case 1: The model was saved correctly (has base_metric)
        if "base_metric" in config:
            config["base_metric"] = tf.keras.metrics.deserialize(config["base_metric"])
            return cls(**config)

        # Case 2: RESCUE MODE - The model is broken (missing base_metric)
        # We infer the correct base metric using the saved 'name' string.
        name = config.get("name", "")
        base_metric = None

        if "expected_count_mae" in name:
            base_metric = ExpectedCountMAE()
        elif "emd" in name:
            base_metric = EMD()
        elif "recall" in name:
            base_metric = ExpectedRecall()
        elif "precision" in name:
            base_metric = ExpectedPrecision()
        elif "binary_accuracy" in name:
            base_metric = ExpectedBinaryAccuracy()
        
        if base_metric is None:
            # Fallback for unexpected names
            print(f"Warning: Could not infer base_metric for {name}. Defaulting to ExpectedCountMAE.")
            base_metric = ExpectedCountMAE()

        # Inject the recovered metric and return the class
        return cls(base_metric=base_metric, **config)


# ------------------------------------------------------------
# KL divergence as a metric for slice models
# ------------------------------------------------------------

class SliceToCountKLDivergence(tf.keras.metrics.Metric):
    """
    KL divergence between soft count distributions derived from slice logits.
    """

    def __init__(self, max_bin=4, name="slice2count_kl", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_bin = max_bin
        self.kl = tf.keras.losses.KLDivergence()

        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_counts = targets_to_soft_counts(y_true, max_bin=self.max_bin)
        pred_counts = slices_to_soft_counts(y_pred, max_bin=self.max_bin)

        value = self.kl(true_counts, pred_counts)

        self.total.assign_add(tf.reduce_sum(value))
        self.count.assign_add(tf.cast(tf.size(value), tf.float32))

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


# ------------------------------------------------------------
# Wrapped versions of your existing clip-level metrics
# ------------------------------------------------------------

def SliceExpectedCountMAE(max_bin=4):
    return SliceToCountWrapper(
        base_metric=ExpectedCountMAE(),
        max_bin=max_bin,
        name="slice2count_expected_count_mae"
    )

def SliceEMD(max_bin=4):
    return SliceToCountWrapper(
        base_metric=EMD(),
        max_bin=max_bin,
        name="slice2count_emd"
    )

def SliceExpectedRecall(max_bin=4):
    return SliceToCountWrapper(
        base_metric=ExpectedRecall(),
        max_bin=max_bin,
        name="slice2count_expected_recall"
    )

def SliceExpectedPrecision(max_bin=4):
    return SliceToCountWrapper(
        base_metric=ExpectedPrecision(),
        max_bin=max_bin,
        name="slice2count_expected_precision"
    )

def SliceExpectedBinaryAccuracy(max_bin=4):
    return SliceToCountWrapper(
        base_metric=ExpectedBinaryAccuracy(),
        max_bin=max_bin,
        name="slice2count_expected_binary_accuracy"
    )


# ------------------------------------------------------------
# Loss terms
# ------------------------------------------------------------

def soft_count_kl_loss(y_true_slices, y_pred_logits, max_bin=4):
    """
    y_true_slices: [B, T] slice-level soft labels
    y_pred_logits: [B, T] logits
    returns: scalar KL loss
    """

    # Convert to soft count distributions
    true_counts = targets_to_soft_counts(y_true_slices, max_bin=max_bin)   # [B, K]
    pred_counts = slices_to_soft_counts(y_pred_logits, max_bin=max_bin)    # [B, K]

    # Numerical stability
    eps = 1e-7
    true_counts = tf.clip_by_value(true_counts, eps, 1.0)
    pred_counts = tf.clip_by_value(pred_counts, eps, 1.0)

    # KL divergence: sum p * log(p/q)
    kl = tf.reduce_sum(true_counts * tf.math.log(true_counts / pred_counts), axis=-1)  # [B]

    return tf.reduce_mean(kl)

class SliceLossWithSoftCountKL(tf.keras.losses.Loss):
    def __init__(self, max_bin=4, kl_weight=1.0, name="slice_bce_plus_kl", **kwargs):
        super().__init__(name=name)
        self.max_bin = max_bin
        self.kl_weight = kl_weight
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def call(self, y_true, y_pred_logits):
        # Slice-level BCE
        bce_loss = self.bce(y_true, y_pred_logits)

        # Soft-count KL
        kl_loss = soft_count_kl_loss(y_true, y_pred_logits, max_bin=self.max_bin)

        return bce_loss + self.kl_weight * kl_loss