"""
metrics.py

Custom losses and metrics for the Frog Perch downstream model.
"""
import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="frog_perch")
class AnnealedLossWrapper(tf.keras.losses.Loss):
    def __init__(self, base_loss, weight_var, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.base_loss = base_loss
        self.weight_var = weight_var

    def call(self, y_true, y_pred):
        # Multiplication happens in the graph
        # For evaluation, weight_var is typically a constant float
        return self.base_loss(y_true, y_pred) * self.weight_var

    def get_config(self):
        """Standard Keras method to enable model saving."""
        config = super().get_config()
        config.update({
            "base_loss": tf.keras.losses.serialize(self.base_loss),
            # We save the current value of the variable
            "weight_var": float(tf.keras.backend.get_value(self.weight_var)) 
                          if hasattr(self.weight_var, 'numpy') or isinstance(self.weight_var, tf.Variable) 
                          else self.weight_var
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Standard Keras method to enable model loading."""
        base_loss_config = config.pop("base_loss")
        # Deserialize the base loss (e.g., BinaryCrossentropy)
        base_loss = tf.keras.losses.deserialize(base_loss_config)
        weight_var = config.pop("weight_var")
        return cls(base_loss=base_loss, weight_var=weight_var, **config)

# Diagnostics can be uncommented, but only work if model is compiled in eager mode.
@tf.keras.utils.register_keras_serializable(package="frog_perch")
class NormalizedEarthMoversDistance1D(tf.keras.losses.Loss):
    """Computes the normalized 1D EMD between ordinal distributions."""
    
    def __init__(self, name="emd_loss", **kwargs):
        super().__init__(name=name, **kwargs)

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # # Troubleshooting/Diagnostics
        # # 1. Shape Check (Look for that trailing 1)
        # tf.print("\n[SHAPE] y_true:", tf.shape(y_true), "| y_pred:", tf.shape(y_pred))
        
        # # 2. Sum Check (Must be 1.0)
        # true_sums = tf.reduce_sum(y_true, axis=-1)
        # pred_sums = tf.reduce_sum(y_pred, axis=-1)
        # tf.print("[SUMS]  y_true avg:", tf.reduce_mean(true_sums), 
        #          "| y_pred avg:", tf.reduce_mean(pred_sums))
        
        # # 3. Validity Check (Look for negatives or NaNs)
        # min_pred = tf.reduce_min(y_pred)
        # max_pred = tf.reduce_max(y_pred)
        # tf.print("[RANGE] y_pred min:", min_pred, "| max:", max_pred)
        # # --------------------------------
        
        cdf_true = tf.cumsum(y_true, axis=-1)
        cdf_pred = tf.cumsum(y_pred, axis=-1)
        
        return tf.reduce_mean(tf.abs(cdf_true - cdf_pred), axis=-1)


@tf.keras.utils.register_keras_serializable(package="frog_perch")
class ExpectedCountMAE(tf.keras.metrics.Metric):
    """Computes the Mean Absolute Error between the expected counts of two distributions."""
    
    def __init__(self, max_bin=16, name="count_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.max_bin = max_bin
        self.total_mae = self.add_weight(name="total_mae", initializer="zeros")
        self.count_probs = self.add_weight(name="count_probs", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        
        # FIX: Use static range instead of dynamic tf.shape to survive Keras graph tracing
        bins = tf.range(self.max_bin + 1, dtype=tf.float32)

        expected_true = tf.reduce_sum(bins * y_true, axis=-1)
        expected_pred = tf.reduce_sum(bins * y_pred, axis=-1)

        mae = tf.abs(expected_true - expected_pred)

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, tf.float32)
            mae = tf.multiply(mae, sample_weight)
            self.count_probs.assign_add(tf.reduce_sum(sample_weight))
        else:
            self.count_probs.assign_add(tf.cast(tf.shape(mae)[0], tf.float32))

        self.total_mae.assign_add(tf.reduce_sum(mae))

    def result(self):
        return tf.math.divide_no_nan(self.total_mae, self.count_probs)

    def reset_state(self):
        self.total_mae.assign(0.0)
        self.count_probs.assign(0.0)