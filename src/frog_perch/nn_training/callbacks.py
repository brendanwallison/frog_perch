"""
callbacks.py

Custom Keras callbacks for the Frog Perch training pipeline.
"""
import sys
import tensorflow as tf

class GPUMemoryCallback(tf.keras.callbacks.Callback):
    """Logs GPU memory usage at the end of each epoch."""
    
    def on_epoch_end(self, epoch: int, logs: dict = None) -> None:
        try:
            info = tf.config.experimental.get_memory_info('GPU:0')
            used = info['current'] / (1024 ** 2)
            peak = info['peak'] / (1024 ** 2)
            sys.stdout.write(f"\r[GPU MEMORY] Epoch {epoch+1}: used={used:.0f} MB | peak={peak:.0f} MB   ")
            sys.stdout.flush()
        except Exception:
            pass


class LossAnnealingCallback(tf.keras.callbacks.Callback):
    """Linearly decays binary and slice loss weights over the course of training."""
    
    def __init__(self, weight_binary: tf.Variable, weight_slice: tf.Variable, total_epochs: int):
        super().__init__()
        self.weight_binary = weight_binary
        self.weight_slice = weight_slice
        self.total_epochs = float(total_epochs)
        
        self.start_binary = 0.1
        self.end_binary = 0.0
        
        self.start_slice = 1.0
        self.end_slice = 0.0

    def on_epoch_begin(self, epoch: int, logs: dict = None) -> None:
        progress = epoch / self.total_epochs
        
        new_bin = self.start_binary - progress * (self.start_binary - self.end_binary)
        new_slice = self.start_slice - progress * (self.start_slice - self.end_slice)
        
        self.weight_binary.assign(max(new_bin, self.end_binary))
        self.weight_slice.assign(max(new_slice, self.end_slice))
        
        print(f"\n[ANNEAL] binary_weight: {new_bin:.4f} | slice_weight: {new_slice:.4f}")