# models/perch_wrapper.py
import tensorflow as tf
import numpy as np

class PerchWrapper:
    """
    Load Perch v2 SavedModel and extract spatial_embedding.
    Expects 1-D numpy waveform of length PERCH_CLIP_SAMPLES at 32000 Hz.

    Usage:
        w = PerchWrapper('/path/to/saved_model')
        spat = w.get_spatial_embedding(waveform)  # returns numpy (16,4,1536)
    """

    def __init__(self, saved_model_path):
        self.saved_model_path = saved_model_path
        self.model = None
        self.signature = None
        self.input_key = None
        self._load()

    def _load(self):
        if self.saved_model_path is None:
            raise ValueError("PERCH saved_model path not provided.")
        try:
            self.model = tf.saved_model.load(self.saved_model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Perch SavedModel at {self.saved_model_path}: {e}")

        # Try to capture serving_default signature
        if hasattr(self.model, 'signatures') and 'serving_default' in self.model.signatures:
            self.signature = self.model.signatures['serving_default']
            # infer input key if possible
            input_keys = list(self.signature.structured_input_signature[1].keys())
            self.input_key = input_keys[0] if input_keys else None
        else:
            self.signature = None
            self.input_key = None

    def _call(self, waveform_batch):
        """
        waveform_batch: numpy array shape [B, S], dtype float32
        Returns:
            dict-like outputs (tf tensors) or tensor
        """
        t = tf.convert_to_tensor(waveform_batch, dtype=tf.float32)
        if self.signature is not None:
            if self.input_key is not None:
                return self.signature(**{self.input_key: t})
            else:
                return self.signature(t)
        else:
            return self.model(t)

    def get_spatial_embedding(self, waveform):
        """
        waveform: 1-D numpy array (S,) or (1,S)
        returns: numpy array (16,4,1536)
        """
        single = False
        if waveform.ndim == 1:
            waveform = waveform[None, :]
            single = True

        outputs = self._call(waveform)

        # If outputs is a dict-like mapping, search for spatial keys
        if isinstance(outputs, dict):
            # common key names
            keys = list(outputs.keys())
            for key in ['spatial_embedding', 'spatial_embeddings', 'spatial_emb', 'spatial_embs', 'spatial']:
                if key in outputs:
                    arr = outputs[key].numpy()
                    return arr[0] if single else arr
            # try any key containing 'spatial' or 'prepool'
            for key in keys:
                if 'spatial' in key.lower() or 'prepool' in key.lower() or 'spatial_emb' in key.lower():
                    arr = outputs[key].numpy()
                    return arr[0] if single else arr
            # if we get here, no spatial key found -> helpful debugging info
            print("Perch wrapper: available output keys:", keys)
            raise RuntimeError("Perch SavedModel did not expose a 'spatial_embedding' key. Inspect above keys and adapt the wrapper.")
        else:
            # outputs may be a tensor -> likely pooled embedding only
            raise RuntimeError("Perch SavedModel returned a tensor (likely pooled embedding) — spatial_embedding not found.")
