import numpy as np
from frog_perch.models.perch_wrapper import PerchWrapper

def main():
    perch = PerchWrapper()

    # Make a batch of random waveforms
    batch_size = 4
    num_samples = 5 * 32000  # 5 seconds @ 32kHz
    waveforms = np.random.uniform(-1, 1, (batch_size, num_samples)).astype(np.float32)

    # Get Perch embeddings
    emb = perch.get_embedding(waveforms)                # [B, 1536]
    spatial = perch.get_spatial_embedding(waveforms)    # [B, H, W, 1536]
    print("spatial shape:", spatial.shape)

    # Mean across spatial dimensions
    mean_spatial = spatial.mean(axis=(1, 2))            # [B, 1536]

    # Compare numerically
    diff = np.abs(mean_spatial - emb)
    print("Max abs diff:", diff.max())
    print("Mean abs diff:", diff.mean())

    # Check exact (within FP tolerance)
    same = np.allclose(mean_spatial, emb, atol=1e-6)
    print("Do they match? ", same)

    # Print an example slice
    print("Example emb[:5]:", emb[0, :5])
    print("Example mean_spatial[:5]:", mean_spatial[0, :5])

if __name__ == "__main__":
    main()
