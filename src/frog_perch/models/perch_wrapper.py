# models/perch_wrapper.py
from frog_perch.utils.tf_env import *  # ensures env vars set before TF import

import tensorflow as tf
from pathlib import Path
from frog_perch.utils.download_perch import get_perch_savedmodel_path


class PerchWrapper:
    """
    Loads Perch v2 SavedModel from KaggleHub's managed cache.
    Uses XLA-disabled execution to avoid CUDA-only module crashes.
    """

    def __init__(self):
        self.saved_model_path = Path(get_perch_savedmodel_path())
        self.model = None
        self.signature = None
        self.input_key = None
        self._load()

    def _load(self):
        try:
            self.model = tf.saved_model.load(str(self.saved_model_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Perch SavedModel at {self.saved_model_path}:\n{e}"
            )

        # Load serving signature if available
        if hasattr(self.model, "signatures") and "serving_default" in self.model.signatures:
            self.signature = self.model.signatures["serving_default"]
            sig_inputs = self.signature.structured_input_signature[1]
            self.input_key = next(iter(sig_inputs.keys()))
        else:
            self.signature = None
            self.input_key = None

    def _call(self, waveform_batch):
        t = tf.convert_to_tensor(waveform_batch, dtype=tf.float32)

        if self.signature is not None:
            return self.signature(**{self.input_key: t})
        else:
            return self.model(t)

    def get_spatial_embedding(self, waveform):
        single = False
        if waveform.ndim == 1:
            waveform = waveform[None, :]
            single = True

        out = self._call(waveform)

        if not isinstance(out, dict):
            raise RuntimeError("Expected dict of outputs from Perch SavedModel.")

        # Find the spatial key
        for k in out.keys():
            if "spatial" in k.lower():
                arr = out[k].numpy()
                return arr[0] if single else arr

        raise RuntimeError(f"No spatial embedding key in outputs: {list(out.keys())}")
