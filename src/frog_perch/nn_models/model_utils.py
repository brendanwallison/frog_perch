"""
model_utils.py

Shared utilities for loading and saving custom Keras models.
"""
import tensorflow as tf

def load_custom_model(ckpt_path: str) -> tf.keras.Model:
    """
    Loads the downstream model from disk, injecting all required custom 
    layers, losses, and metrics into the Keras deserialization process.
    """
    # Import locally to avoid circular dependencies if this file gets imported early
    from frog_perch.nn_models.downstream import SoftCountFromSlices
    from frog_perch.nn_training.metrics import (
        AnnealedLossWrapper, 
        ExpectedCountMAE, 
        NormalizedEarthMoversDistance1D
    )

    custom_objects = {
        "SoftCountFromSlices": SoftCountFromSlices,
        "AnnealedLossWrapper": AnnealedLossWrapper,
        "ExpectedCountMAE": ExpectedCountMAE,
        "NormalizedEarthMoversDistance1D": NormalizedEarthMoversDistance1D,
    }
    
    return tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects, compile=False)