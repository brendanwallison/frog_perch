# main.py
from frog_perch.training.train import train
import frog_perch.config as config

# diagnostics.py
import os
# ensure you run this in the same shell that launches your app
print("ENV TF_XLA_FLAGS:", os.environ.get("TF_XLA_FLAGS"))
print("ENV XLA_FLAGS:", os.environ.get("XLA_FLAGS"))
print("ENV TF_CPP_MIN_LOG_LEVEL:", os.environ.get("TF_CPP_MIN_LOG_LEVEL"))

import tensorflow as tf
print("TF version:", tf.__version__)
print("tf.test.is_built_with_cuda():", tf.test.is_built_with_cuda())
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))
print("Logical GPUs:", tf.config.list_logical_devices("GPU") if tf.config.list_physical_devices("GPU") else "none")
# JIT / optimizer state
try:
    print("JIT enabled:", tf.config.optimizer.get_jit())
except Exception:
    pass


if __name__ == '__main__':
    train(label_mode=config.LABEL_MODE, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
