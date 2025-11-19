import tensorflow as tf

# ============================================================
#   Metric 1: Binary Presence Accuracy using Expected Count
# ============================================================

class ExpectedRecall(tf.keras.metrics.Metric):
    def __init__(self, name="expected_recall", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fn = self.add_weight(name="fn", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(tf.argmax(y_true, axis=-1) > 0, tf.float32)

        bins = tf.cast(tf.range(tf.shape(y_pred)[-1]), tf.float32)
        exp_pred = tf.reduce_sum(y_pred * bins, axis=-1)
        y_pred_bin = tf.cast(exp_pred >= 0.5, tf.float32)

        tp = tf.reduce_sum(tf.cast(y_pred_bin * y_true_bin, tf.float32))
        fn = tf.reduce_sum(tf.cast((1 - y_pred_bin) * y_true_bin, tf.float32))

        self.tp.assign_add(tp)
        self.fn.assign_add(fn)

    def result(self):
        return self.tp / (self.tp + self.fn + 1e-8)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fn.assign(0.0)

class ExpectedPrecision(tf.keras.metrics.Metric):
    def __init__(self, name="expected_precision", **kwargs):
        super().__init__(name=name, **kwargs)
        self.tp = self.add_weight(name="tp", shape=(), initializer="zeros")
        self.fp = self.add_weight(name="fp", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_bin = tf.cast(tf.argmax(y_true, axis=-1) > 0, tf.float32)

        bins = tf.cast(tf.range(tf.shape(y_pred)[-1]), tf.float32)
        exp_pred = tf.reduce_sum(y_pred * bins, axis=-1)
        y_pred_bin = tf.cast(exp_pred >= 0.5, tf.float32)

        tp = tf.reduce_sum(tf.cast(y_pred_bin * y_true_bin, tf.float32))
        fp = tf.reduce_sum(tf.cast(y_pred_bin * (1 - y_true_bin), tf.float32))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)

    def result(self):
        return self.tp / (self.tp + self.fp + 1e-8)

    def reset_states(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)


class ExpectedBinaryAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="expected_binary_accuracy", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", shape=(), initializer="zeros")
        self.total = self.add_weight(name="total", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # true binary (any-count >0)
        y_true_bin = tf.cast(tf.argmax(y_true, axis=-1) > 0, tf.float32)

        # predicted binary from expected count >= 0.5
        bins = tf.cast(tf.range(tf.shape(y_pred)[-1]), tf.float32)
        exp_pred = tf.reduce_sum(y_pred * bins, axis=-1)
        y_pred_bin = tf.cast(exp_pred >= 0.5, tf.float32)

        correct = tf.reduce_sum(tf.cast(tf.equal(y_true_bin, y_pred_bin), tf.float32))
        total = tf.cast(tf.size(y_true_bin), tf.float32)

        self.correct.assign_add(correct)
        self.total.assign_add(total)

    def result(self):
        return self.correct / (self.total + 1e-8)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

# ============================================================
#   Metric 2: Binary Presence Accuracy using Argmax (0-count bin)
# ============================================================

class CountPresenceAccuracyArgmax(tf.keras.metrics.Metric):
    """
    Binary presence/absence accuracy computed by:
        presence = argmax != 0-bin
        (bin 0 is assumed to represent "no individuals")
    """

    def __init__(self, name="count_presence_acc_argmax", **kwargs):
        super().__init__(name=name, **kwargs)
        self.correct = self.add_weight(name="correct", initializer="zeros")
        self.total = self.add_weight(name="total", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        # hard classes
        true_argmax = tf.argmax(y_true, axis=-1)
        pred_argmax = tf.argmax(y_pred, axis=-1)

        true_presence = tf.cast(true_argmax > 0, tf.int32)
        pred_presence = tf.cast(pred_argmax > 0, tf.int32)

        matches = tf.cast(tf.equal(true_presence, pred_presence), tf.float32)

        if sample_weight is not None:
            matches *= tf.cast(sample_weight, tf.float32)

        self.correct.assign_add(tf.reduce_sum(matches))
        self.total.assign_add(tf.cast(tf.size(matches), tf.float32))

    def result(self):
        return tf.math.divide_no_nan(self.correct, self.total)

    def reset_states(self):
        self.correct.assign(0.0)
        self.total.assign(0.0)

class EMD(tf.keras.metrics.Metric):
    def __init__(self, name="emd", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")


    def update_state(self, y_true, y_pred, sample_weight=None):
        # cumulative distribution functions
        cdf_true = tf.cumsum(y_true, axis=-1)
        cdf_pred = tf.cumsum(y_pred, axis=-1)
        dist = tf.reduce_sum(tf.abs(cdf_true - cdf_pred), axis=-1)
        self.total.assign_add(tf.reduce_sum(dist))
        self.count.assign_add(tf.cast(tf.size(dist), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class ExpectedCountMAE(tf.keras.metrics.Metric):
    def __init__(self, name="exp_count_mae", **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        bins = tf.cast(tf.range(tf.shape(y_pred)[-1]), tf.float32)
        exp_true = tf.reduce_sum(y_true * bins, axis=-1)
        exp_pred = tf.reduce_sum(y_pred * bins, axis=-1)
        mae = tf.abs(exp_true - exp_pred)
        self.total.assign_add(tf.reduce_sum(mae))
        self.count.assign_add(tf.cast(tf.size(mae), tf.float32))

    def result(self):
        return self.total / (self.count + 1e-8)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
